#!/usr/bin/env python3
"""
REGIME ACID TEST v3 — COMPREHENSIVE STRESS TEST
=================================================
Same trade generation as acid test v1, plus:
  1. Spread at entry captured for every trade
  2. Multiple spread filter thresholds tested
  3. Rollover window exclusion
  4. Slippage simulation (0, 0.5, 1, 1.5, 2, 3 pips random adverse fill)
  5. Consecutive loss streak analysis
  6. Max drawdown duration (days without new equity high)
  7. Pair-specific adaptive spread thresholds (P75 per pair)
  8. Correlation clustering (concurrent entries per M5 bar)

Run: python regime_acid_test_v3_spread.py --data-dir "C:\Forex_Projects\5_year_5s_data"
"""

import os, sys, glob, argparse, time as time_mod, logging
import numpy as np
import pandas as pd

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIG — FIXED. IDENTICAL TO ACID TEST v1.
# ══════════════════════════════════════════════════════════════════════

ENTRY_THRESHOLD = 0.4
EXIT_THRESHOLD  = 0.2
ENTRY_CONFIRM   = 0
EXIT_CONFIRM    = 0
TIMED_EXIT      = 576    # 48h max hold

MTF_WEIGHTS = {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}

ALL_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
             'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
             'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
             'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD']

SPREAD_FILTERS = [0, 3, 5, 8, 10, 15]
ROLLOVER_WINDOWS = {41, 42, 43}  # 20:30-22:00 UTC
SLIPPAGE_LEVELS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]  # pips

# For reproducible slippage simulation
RNG_SEED = 42


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_pair_data(filepath):
    df = pd.read_parquet(filepath)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time')
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    for col in ['open', 'high', 'low', 'close']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def build_timeframes(df_5s):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df_5s.columns:
        agg['volume'] = 'sum'
    has_ba = all(c in df_5s.columns for c in ['bid_high', 'bid_low', 'bid_close',
                                                'ask_high', 'ask_low', 'ask_close'])
    if has_ba:
        for c, f in {'bid_open': 'first', 'bid_high': 'max', 'bid_low': 'min', 'bid_close': 'last',
                      'ask_open': 'first', 'ask_high': 'max', 'ask_low': 'min', 'ask_close': 'last'}.items():
            if c in df_5s.columns:
                agg[c] = f
    tfs = {}
    for label, rule in [('M1','1min'),('M5','5min'),('M15','15min'),('H1','1h'),('H4','4h')]:
        df = df_5s.resample(rule).agg(agg).dropna(subset=['close'])
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        tfs[label] = df
    return tfs


# ══════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════

def _ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr); out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

def add_ema(df):
    if len(df) < 22: return df
    c = df['close'].values.astype(np.float64)
    if HAS_TALIB:
        df['ema_9'] = talib.EMA(c, timeperiod=9)
        df['ema_21'] = talib.EMA(c, timeperiod=21)
    else:
        df['ema_9'] = _ema_np(c, 9); df['ema_21'] = _ema_np(c, 21)
    return df


# Bar durations in nanoseconds — higher-TF signals are only available AFTER
# the bar closes.  Shifting timestamps forward by this amount prevents the
# backtest from using a bar whose close price is still in the future.
_TF_PERIOD_NS = {
    'M1':  60_000_000_000,           #  1 min
    'M5':  300_000_000_000,          #  5 min
    'M15': 900_000_000_000,          # 15 min
    'H1':  3_600_000_000_000,        #  1 hour
    'H4':  14_400_000_000_000,       #  4 hours
}


def compute_mtf_bias(tfs, m5_index):
    total_w = sum(MTF_WEIGHTS.values())
    n = len(m5_index); bias = np.zeros(n, dtype=np.float64)
    tf_data = {}
    for tf_name in MTF_WEIGHTS:
        df = tfs.get(tf_name)
        if df is None or len(df) < 22 or 'ema_9' not in df.columns: continue
        e9 = df['ema_9'].values; e21 = df['ema_21'].values; cl = df['close'].values
        sig = np.where((e9>e21)&(cl>e9), 1.0, np.where((e9<e21)&(cl<e9), -1.0, 0.0))
        # FIX: shift higher-TF timestamps forward by bar period so a bar's
        # signal is only available AFTER the bar closes.  Without this, the
        # M15/H1/H4 signal at M5 bar 10:05 uses the close from 10:14:55 /
        # 10:59:55 / 11:59:55 — pure look-ahead bias.
        # M5 is the base TF (we evaluate at its close) — no shift needed.
        # M1 completes within each M5 bar — no shift needed.
        tt = df.index.asi8
        if tf_name in ('M15', 'H1', 'H4'):
            tt = tt + _TF_PERIOD_NS[tf_name]
        tf_data[tf_name] = (tt, sig)
    m5_ns = m5_index.asi8
    for tf_name, w in MTF_WEIGHTS.items():
        if tf_name not in tf_data: continue
        tt, ts = tf_data[tf_name]
        idx = np.clip(np.searchsorted(tt, m5_ns, side='right') - 1, 0, len(ts) - 1)
        bias += ts[idx] * w
    if total_w > 0 and total_w != 1.0: bias /= total_w
    return np.clip(bias, -1.0, 1.0)


def _regime_py(bias, et, xt):
    n = len(bias); state = np.zeros(n, dtype=np.int32); cur = 0
    for i in range(n):
        b = bias[i]
        if cur == 0:
            if b > et: cur = 1
            elif b < -et: cur = -1
        elif cur == 1:
            if b < xt:
                cur = 0
                if b < -et: cur = -1
        else:
            if b > -xt:
                cur = 0
                if b > et: cur = 1
        state[i] = cur
    return state

if HAS_NUMBA:
    @njit(cache=True)
    def regime_hysteresis(bias, et, xt):
        n = len(bias); state = np.zeros(n, dtype=np.int32); cur = np.int32(0)
        for i in range(n):
            b = bias[i]
            if cur == 0:
                if b > et: cur = np.int32(1)
                elif b < -et: cur = np.int32(-1)
            elif cur == 1:
                if b < xt:
                    cur = np.int32(0)
                    if b < -et: cur = np.int32(-1)
            else:
                if b > -xt:
                    cur = np.int32(0)
                    if b > et: cur = np.int32(1)
            state[i] = cur
        return state
else:
    regime_hysteresis = _regime_py


# ══════════════════════════════════════════════════════════════════════
# TRADE GENERATION — WITH SPREAD CAPTURE
# ══════════════════════════════════════════════════════════════════════

def generate_trades(state, m5_index, bid_c, ask_c, pair_name, pip_mult):
    n = len(state)
    m5_ns = m5_index.asi8
    trades = []

    for i in range(1, n):
        if state[i-1] == 0 and state[i] != 0:
            ts = m5_index[i]
            if ts.dayofweek >= 5:
                continue

            d = int(state[i])
            entry_bar = i + ENTRY_CONFIRM

            if entry_bar >= n:
                continue
            if state[entry_bar] != d:
                continue

            entry_ask = ask_c[entry_bar]
            entry_bid = bid_c[entry_bar]
            entry_spread_pips = (entry_ask - entry_bid) * pip_mult
            entry_price = entry_ask if d == 1 else entry_bid
            entry_ns = m5_ns[entry_bar]

            # Find exit bar
            exit_bar = min(entry_bar + TIMED_EXIT, n - 1)
            for j in range(entry_bar + 1, min(entry_bar + TIMED_EXIT + 1, n)):
                if state[j] != d:
                    exit_bar = j
                    break

            exit_price = bid_c[exit_bar] if d == 1 else ask_c[exit_bar]
            exit_ns = m5_ns[exit_bar]

            pnl = d * (exit_price - entry_price) * pip_mult
            hold_bars = exit_bar - entry_bar

            window = ts.hour * 2 + (1 if ts.minute >= 30 else 0)

            trades.append({
                'pair': pair_name,
                'direction': d,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_ns': entry_ns,
                'exit_ns': exit_ns,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_spread_pips': entry_spread_pips,
                'pnl_pips': pnl,
                'hold_bars': hold_bars,
                'hold_hours': hold_bars * 5 / 60,
                'year': ts.year,
                'dow': ts.dayofweek,
                'window': window,
            })

    return trades


def apply_position_blocking(trades_df):
    trades_df = trades_df.sort_values('entry_ns').reset_index(drop=True)
    last_exit = {}
    keep = []
    for idx, row in trades_df.iterrows():
        pair = row['pair']
        if pair in last_exit and row['entry_ns'] < last_exit[pair]:
            continue
        keep.append(idx)
        last_exit[pair] = row['exit_ns']
    return trades_df.loc[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ══════════════════════════════════════════════════════════════════════

def analyse_filtered(df, label):
    n = len(df)
    if n == 0:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0, 'wr': 0, 'pf': 0, 'dd': 0}
    pnl = df['pnl_pips'].values
    total = pnl.sum()
    avg = pnl.mean()
    wr = (pnl > 0).mean() * 100
    gl = abs(pnl[pnl < 0].sum())
    pf = pnl[pnl > 0].sum() / gl if gl > 0 else 999
    cum = np.cumsum(pnl)
    dd = (np.maximum.accumulate(cum) - cum).max()
    return {
        'label': label, 'n': n, 'total': total, 'avg': avg,
        'wr': wr, 'pf': pf, 'dd': dd,
    }


def print_comparison(results, title="FILTER COMPARISON"):
    w = 115
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)
    print(f"\n  {'Config':<40} {'Trades':>8} {'Total Pips':>14} {'Avg':>8} {'WR':>6} {'PF':>6} {'Max DD':>10}")
    print("  " + "-" * 105)
    for r in results:
        if r['n'] == 0:
            print(f"  {r['label']:<40} {'0':>8}")
            continue
        print(f"  {r['label']:<40} {r['n']:>8,} {r['total']:>+14,.1f} {r['avg']:>+8.3f} "
              f"{r['wr']:>5.1f}% {r['pf']:>6.3f} {r['dd']:>10,.0f}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 1: SLIPPAGE SIMULATION
# ══════════════════════════════════════════════════════════════════════

def run_slippage_analysis(blocked_df):
    """
    Simulate random adverse slippage at entry.
    For each trade, slippage = uniform(0, max_slip) pips AGAINST you.
    Long: entry price increases by slippage -> pnl decreases
    Short: entry price decreases by slippage -> pnl decreases
    i.e. slippage ALWAYS hurts, never helps (conservative).
    """
    print(f"\n{'='*115}")
    print(f"  SLIPPAGE SIMULATION (adverse fill, 10 runs per level, seed={RNG_SEED})")
    print(f"{'='*115}")

    base_pnl = blocked_df['pnl_pips'].values
    n_trades = len(base_pnl)
    base_total = base_pnl.sum()
    base_avg = base_pnl.mean()

    print(f"\n  Baseline: {n_trades:,} trades, {base_total:+,.1f} total pips, {base_avg:+.3f} avg")

    print(f"\n  {'Slippage':>10} {'Avg Total':>14} {'Avg/Trade':>10} {'D Total':>12} {'D %':>8} "
          f"{'WR':>6} {'PF':>6} {'Worst DD':>12}")
    print("  " + "-" * 95)

    rng = np.random.RandomState(RNG_SEED)
    N_RUNS = 10

    for slip in SLIPPAGE_LEVELS:
        if slip == 0:
            cum = np.cumsum(base_pnl)
            dd = (np.maximum.accumulate(cum) - cum).max()
            wr = (base_pnl > 0).mean() * 100
            gl = abs(base_pnl[base_pnl < 0].sum())
            pf = base_pnl[base_pnl > 0].sum() / gl if gl > 0 else 999
            print(f"  {slip:>9.1f}p {base_total:>+14,.1f} {base_avg:>+10.3f} {'---':>12} {'---':>8} "
                  f"{wr:>5.1f}% {pf:>6.3f} {dd:>12,.0f}")
            continue

        run_totals = []
        run_avgs = []
        run_wrs = []
        run_pfs = []
        run_dds = []

        for run in range(N_RUNS):
            # Random adverse slippage: uniform(0, slip) per trade
            slip_cost = rng.uniform(0, slip, size=n_trades)
            degraded_pnl = base_pnl - slip_cost

            run_totals.append(degraded_pnl.sum())
            run_avgs.append(degraded_pnl.mean())
            run_wrs.append((degraded_pnl > 0).mean() * 100)
            gl = abs(degraded_pnl[degraded_pnl < 0].sum())
            pf = degraded_pnl[degraded_pnl > 0].sum() / gl if gl > 0 else 999
            run_pfs.append(pf)
            cum = np.cumsum(degraded_pnl)
            dd = (np.maximum.accumulate(cum) - cum).max()
            run_dds.append(dd)

        avg_total = np.mean(run_totals)
        avg_avg = np.mean(run_avgs)
        avg_wr = np.mean(run_wrs)
        avg_pf = np.mean(run_pfs)
        worst_dd = max(run_dds)
        delta = avg_total - base_total
        delta_pct = delta / base_total * 100

        print(f"  {slip:>9.1f}p {avg_total:>+14,.1f} {avg_avg:>+10.3f} {delta:>+12,.1f} "
              f"{delta_pct:>+7.1f}% {avg_wr:>5.1f}% {avg_pf:>6.3f} {worst_dd:>12,.0f}")

    # Break-even analysis
    print(f"\n  Break-even slippage (where avg pnl per trade -> 0):")
    print(f"  Avg edge per trade = {base_avg:+.3f} pips")
    print(f"  Uniform(0, X) has mean X/2")
    print(f"  Break-even: X/2 = {base_avg:.3f} -> X = {base_avg * 2:.1f} pips")
    print(f"  System survives up to ~{base_avg * 2:.1f} pips MAX adverse slippage per trade")
    print(f"  (at avg slippage of {base_avg:.1f} pips per trade)")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 2: CONSECUTIVE LOSS STREAKS
# ══════════════════════════════════════════════════════════════════════

def run_streak_analysis(blocked_df):
    print(f"\n{'='*115}")
    print(f"  CONSECUTIVE LOSS STREAK ANALYSIS")
    print(f"{'='*115}")

    df = blocked_df.sort_values('entry_ns').reset_index(drop=True)
    pnl = df['pnl_pips'].values
    is_loss = pnl < 0

    # Find all loss streaks
    streaks = []
    current_streak = 0
    streak_start_idx = 0
    streak_pnl = 0.0

    for i in range(len(is_loss)):
        if is_loss[i]:
            if current_streak == 0:
                streak_start_idx = i
                streak_pnl = 0.0
            current_streak += 1
            streak_pnl += pnl[i]
        else:
            if current_streak > 0:
                streaks.append({
                    'length': current_streak,
                    'pnl': streak_pnl,
                    'start_idx': streak_start_idx,
                    'end_idx': i - 1,
                })
            current_streak = 0

    if current_streak > 0:
        streaks.append({
            'length': current_streak,
            'pnl': streak_pnl,
            'start_idx': streak_start_idx,
            'end_idx': len(is_loss) - 1,
        })

    if not streaks:
        print("  No loss streaks found (?!)")
        return

    streak_lengths = [s['length'] for s in streaks]
    streak_pnls = [s['pnl'] for s in streaks]

    print(f"\n  Total loss streaks: {len(streaks):,}")
    print(f"\n  Streak length distribution:")
    print(f"    Mean:   {np.mean(streak_lengths):.1f}")
    print(f"    Median: {np.median(streak_lengths):.0f}")
    print(f"    P90:    {np.percentile(streak_lengths, 90):.0f}")
    print(f"    P95:    {np.percentile(streak_lengths, 95):.0f}")
    print(f"    P99:    {np.percentile(streak_lengths, 99):.0f}")
    print(f"    Max:    {max(streak_lengths)}")

    # Distribution table
    print(f"\n  {'Streak':>8} {'Count':>8} {'Cumulative':>12} {'Worst PnL':>12}")
    bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]
    for b in bins:
        matching = [s for s in streaks if s['length'] >= b]
        if not matching:
            break
        worst = min(s['pnl'] for s in matching)
        print(f"  {b:>7}+ {len(matching):>8,} {len(matching)/len(streaks)*100:>10.1f}% {worst:>+12.1f}")

    # Top 20 worst streaks
    top_streaks = sorted(streaks, key=lambda s: s['pnl'])[:20]
    print(f"\n  TOP 20 WORST LOSS STREAKS:")
    print(f"  {'Rank':>4} {'Length':>8} {'Total PnL':>12} {'Avg/Loss':>10} {'Period':>40}")
    for rank, s in enumerate(top_streaks, 1):
        avg_per = s['pnl'] / s['length']
        start_dt = pd.Timestamp(df.iloc[s['start_idx']]['entry_ns'], unit='ns')
        end_dt = pd.Timestamp(df.iloc[s['end_idx']]['entry_ns'], unit='ns')
        period = f"{start_dt.strftime('%Y-%m-%d %H:%M')} -> {end_dt.strftime('%Y-%m-%d %H:%M')}"
        print(f"  {rank:>4} {s['length']:>8} {s['pnl']:>+12.1f} {avg_per:>+10.1f} {period:>40}")

    # Account impact
    worst = min(streak_pnls)
    worst_len = max(streak_lengths)
    print(f"\n  IMPACT ON GBP200 ACCOUNT (GBP0.01/pip, 19 pairs):")
    print(f"    Worst streak PnL:    {worst:+.1f} pips = GBP{abs(worst) * 0.01:.2f}")
    print(f"    Longest streak:      {worst_len} consecutive losses")

    # Per-pair streaks
    print(f"\n  PER-PAIR WORST STREAKS:")
    print(f"  {'Pair':>10} {'Worst Run':>10} {'PnL':>10} {'Avg/Loss':>10}")
    for pair in sorted(df['pair'].unique()):
        pair_pnl = df[df['pair'] == pair]['pnl_pips'].values
        pair_loss = pair_pnl < 0
        max_run = 0; cur_run = 0; cur_pnl = 0; worst_run_pnl = 0
        for i in range(len(pair_loss)):
            if pair_loss[i]:
                cur_run += 1; cur_pnl += pair_pnl[i]
                if cur_run > max_run:
                    max_run = cur_run; worst_run_pnl = cur_pnl
                elif cur_run == max_run and cur_pnl < worst_run_pnl:
                    worst_run_pnl = cur_pnl
            else:
                cur_run = 0; cur_pnl = 0
        avg_per = worst_run_pnl / max_run if max_run > 0 else 0
        print(f"  {pair:>10} {max_run:>10} {worst_run_pnl:>+10.1f} {avg_per:>+10.1f}")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 3: DRAWDOWN DURATION
# ══════════════════════════════════════════════════════════════════════

def run_drawdown_duration_analysis(blocked_df):
    print(f"\n{'='*115}")
    print(f"  DRAWDOWN DURATION ANALYSIS (time without new equity high)")
    print(f"{'='*115}")

    df = blocked_df.sort_values('entry_ns').reset_index(drop=True)
    pnl = df['pnl_pips'].values
    entry_times = pd.to_datetime(df['entry_ns'].values, unit='ns')

    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)

    # Find drawdown periods
    in_dd = cum < running_max
    dd_periods = []
    dd_start = None

    for i in range(len(in_dd)):
        if in_dd[i] and dd_start is None:
            dd_start = i
        elif not in_dd[i] and dd_start is not None:
            dd_depth = running_max[dd_start] - cum[dd_start:i].min()
            dd_start_time = entry_times[dd_start]
            dd_end_time = entry_times[i]
            dd_duration_days = (dd_end_time - dd_start_time).total_seconds() / 86400
            dd_n_trades = i - dd_start
            dd_periods.append({
                'start_idx': dd_start,
                'end_idx': i,
                'start_time': dd_start_time,
                'end_time': dd_end_time,
                'duration_days': dd_duration_days,
                'depth_pips': dd_depth,
                'n_trades': dd_n_trades,
            })
            dd_start = None

    if dd_start is not None:
        dd_depth = running_max[dd_start] - cum[dd_start:].min()
        dd_start_time = entry_times[dd_start]
        dd_end_time = entry_times[-1]
        dd_duration_days = (dd_end_time - dd_start_time).total_seconds() / 86400
        dd_periods.append({
            'start_idx': dd_start,
            'end_idx': len(cum) - 1,
            'start_time': dd_start_time,
            'end_time': dd_end_time,
            'duration_days': dd_duration_days,
            'depth_pips': dd_depth,
            'n_trades': len(cum) - dd_start,
            'ongoing': True,
        })

    if not dd_periods:
        print("  No drawdown periods found (?!)")
        return

    durations = [d['duration_days'] for d in dd_periods]
    depths = [d['depth_pips'] for d in dd_periods]

    print(f"\n  Total drawdown periods: {len(dd_periods):,}")
    print(f"\n  Duration distribution (days):")
    print(f"    Mean:   {np.mean(durations):.1f}")
    print(f"    Median: {np.median(durations):.1f}")
    print(f"    P90:    {np.percentile(durations, 90):.1f}")
    print(f"    P95:    {np.percentile(durations, 95):.1f}")
    print(f"    P99:    {np.percentile(durations, 99):.1f}")
    print(f"    Max:    {max(durations):.1f}")

    print(f"\n  Depth distribution (pips):")
    print(f"    Mean:   {np.mean(depths):.1f}")
    print(f"    Median: {np.median(depths):.1f}")
    print(f"    P95:    {np.percentile(depths, 95):.1f}")
    print(f"    Max:    {max(depths):.1f}")

    # Top 20 longest drawdowns
    top_duration = sorted(dd_periods, key=lambda d: -d['duration_days'])[:20]
    print(f"\n  TOP 20 LONGEST DRAWDOWNS:")
    print(f"  {'Rank':>4} {'Days':>8} {'Depth':>10} {'Trades':>8} {'Period':>50}")
    for rank, d in enumerate(top_duration, 1):
        ongoing = " (ONGOING)" if d.get('ongoing') else ""
        period = f"{d['start_time'].strftime('%Y-%m-%d')} -> {d['end_time'].strftime('%Y-%m-%d')}{ongoing}"
        print(f"  {rank:>4} {d['duration_days']:>8.1f} {d['depth_pips']:>+10.1f} "
              f"{d['n_trades']:>8,} {period:>50}")

    # Top 10 deepest drawdowns
    top_depth = sorted(dd_periods, key=lambda d: -d['depth_pips'])[:10]
    print(f"\n  TOP 10 DEEPEST DRAWDOWNS:")
    print(f"  {'Rank':>4} {'Depth':>10} {'Days':>8} {'Trades':>8} {'Period':>50}")
    for rank, d in enumerate(top_depth, 1):
        ongoing = " (ONGOING)" if d.get('ongoing') else ""
        period = f"{d['start_time'].strftime('%Y-%m-%d')} -> {d['end_time'].strftime('%Y-%m-%d')}{ongoing}"
        print(f"  {rank:>4} {d['depth_pips']:>10.1f} {d['duration_days']:>8.1f} "
              f"{d['n_trades']:>8,} {period:>50}")

    # Psychological warning levels
    max_dd_depth = max(depths)
    max_dd_days = max(durations)
    print(f"\n  PSYCHOLOGICAL PREP:")
    print(f"    Worst drawdown: {max_dd_depth:.0f} pips over {max_dd_days:.0f} days")
    print(f"    On GBP200 at GBP0.01/pip: GBP{max_dd_depth * 0.01:.2f} ({max_dd_depth * 0.01 / 200 * 100:.1f}% of account)")
    print(f"    You WILL sit through a {max_dd_days:.0f}-day period watching equity go sideways/down")
    print(f"    This is NORMAL. The system recovers every time in the backtest.")

    # How often underwater for >N days?
    print(f"\n  How often you'll be in drawdown:")
    for threshold in [1, 3, 7, 14, 21, 30, 60]:
        count = sum(1 for d in durations if d >= threshold)
        pct = count / len(durations) * 100 if durations else 0
        print(f"    >{threshold:>2} days: {count:>4} times ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 4: PAIR-SPECIFIC SPREAD THRESHOLDS
# ══════════════════════════════════════════════════════════════════════

def run_adaptive_spread_analysis(blocked_df):
    print(f"\n{'='*115}")
    print(f"  PAIR-SPECIFIC ADAPTIVE SPREAD FILTER")
    print(f"{'='*115}")

    # Compute spread percentiles per pair
    pair_stats = {}
    print(f"\n  {'Pair':>10} {'P25':>8} {'P50':>8} {'P75':>8} {'P90':>8} {'Trades':>8}")
    for pair in sorted(blocked_df['pair'].unique()):
        g = blocked_df[blocked_df['pair'] == pair]
        sp = g['entry_spread_pips'].values
        p25, p50, p75, p90 = [np.percentile(sp, p) for p in [25, 50, 75, 90]]
        pair_stats[pair] = {'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90}
        print(f"  {pair:>10} {p25:>8.2f} {p50:>8.2f} {p75:>8.2f} {p90:>8.2f} {len(g):>8,}")

    # Test multiple adaptive thresholds
    configs = {
        'P75': {p: pair_stats[p]['p75'] for p in pair_stats},
        'P90': {p: pair_stats[p]['p90'] for p in pair_stats},
    }

    results = [analyse_filtered(blocked_df, 'No filter (baseline)')]

    for label, thresholds in configs.items():
        mask = blocked_df.apply(
            lambda r: r['entry_spread_pips'] <= thresholds.get(r['pair'], 999), axis=1
        )
        filtered = blocked_df[mask]
        results.append(analyse_filtered(filtered, f'Adaptive {label} per pair'))

        # + no rollover
        mask_nr = mask & (~blocked_df['window'].isin(ROLLOVER_WINDOWS))
        filtered_nr = blocked_df[mask_nr]
        results.append(analyse_filtered(filtered_nr, f'Adaptive {label} + no rollover'))

    print_comparison(results, "ADAPTIVE SPREAD FILTER COMPARISON")

    # Year-by-year for best adaptive config
    best_thresholds = configs['P75']
    mask_best = blocked_df.apply(
        lambda r: r['entry_spread_pips'] <= best_thresholds.get(r['pair'], 999), axis=1
    )
    mask_best_nr = mask_best & (~blocked_df['window'].isin(ROLLOVER_WINDOWS))
    filtered_best = blocked_df[mask_best_nr]

    print(f"\n  YEARLY: Adaptive P75 + no rollover")
    print(f"  {'Year':>6} {'Trades':>8} {'Total':>12} {'Avg':>8} {'WR':>6} {'PF':>6}")
    for yr, g in filtered_best.groupby('year'):
        pnl = g['pnl_pips'].values
        t = pnl.sum(); a = pnl.mean()
        w = (pnl > 0).mean() * 100
        gl = abs(pnl[pnl < 0].sum())
        p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
        marker = "+" if t > 0 else "-"
        print(f"  {yr:>6} {len(g):>8,} {t:>+12,.1f} {a:>+8.3f} {w:>5.1f}% {p:>6.2f}  {marker}")

    # Output pair-specific thresholds for bot config
    print(f"\n  RECOMMENDED BOT CONFIG (P75 pair-specific max spread):")
    print(f"  PAIR_MAX_SPREAD = {{")
    for pair in sorted(best_thresholds.keys()):
        print(f"      '{pair}': {best_thresholds[pair]:.1f},")
    print(f"  }}")

    return best_thresholds


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS 5: CORRELATION / CLUSTERING
# ══════════════════════════════════════════════════════════════════════

def run_correlation_analysis(blocked_df):
    print(f"\n{'='*115}")
    print(f"  ENTRY CORRELATION / CLUSTERING ANALYSIS")
    print(f"{'='*115}")

    df = blocked_df.sort_values('entry_ns').reset_index(drop=True)

    # Group entries by M5 bar (same entry_ns = same bar)
    bar_groups = df.groupby('entry_ns')
    entries_per_bar = bar_groups.size()
    concurrent_entries = entries_per_bar[entries_per_bar > 1]

    print(f"\n  Total unique entry bars: {len(entries_per_bar):,}")
    print(f"  Bars with >1 concurrent entry: {len(concurrent_entries):,} ({len(concurrent_entries)/len(entries_per_bar)*100:.1f}%)")

    if len(concurrent_entries) > 0:
        print(f"\n  Concurrent entries per bar (when >1):")
        print(f"    Mean: {concurrent_entries.mean():.1f}")
        print(f"    P75:  {np.percentile(concurrent_entries, 75):.0f}")
        print(f"    P90:  {np.percentile(concurrent_entries, 90):.0f}")
        print(f"    P95:  {np.percentile(concurrent_entries, 95):.0f}")
        print(f"    Max:  {concurrent_entries.max()}")

    # Distribution table
    max_conc = int(concurrent_entries.max()) if len(concurrent_entries) > 0 else 1
    print(f"\n  {'Concurrent':>12} {'Bars':>8} {'Trades':>8} {'% of Trades':>12} {'Avg PnL/Cluster':>16}")
    for n_conc in range(1, min(20, max_conc + 1)):
        matching_bars = entries_per_bar[entries_per_bar == n_conc]
        n_trades = len(matching_bars) * n_conc
        pct = n_trades / len(df) * 100
        if len(matching_bars) > 0:
            # Get PnL for these clusters
            matching_ns = matching_bars.index
            cluster_trades = df[df['entry_ns'].isin(matching_ns)]
            cluster_pnls = cluster_trades.groupby('entry_ns')['pnl_pips'].sum()
            avg_cluster_pnl = cluster_pnls.mean()
            print(f"  {n_conc:>12} {len(matching_bars):>8,} {n_trades:>8,} {pct:>11.1f}% {avg_cluster_pnl:>+15.2f}")

    # Direction analysis
    print(f"\n  DIRECTION CLUSTERING (when multiple entries on same bar):")
    same_dir_count = 0
    mixed_dir_count = 0
    cluster_pnls_aligned = []
    cluster_pnls_mixed = []

    for ns, group in bar_groups:
        if len(group) <= 1:
            continue
        dirs = group['direction'].values
        pnls = group['pnl_pips'].values
        if np.all(dirs == dirs[0]):
            same_dir_count += 1
            cluster_pnls_aligned.append(pnls.sum())
        else:
            mixed_dir_count += 1
            cluster_pnls_mixed.append(pnls.sum())

    total_clusters = same_dir_count + mixed_dir_count
    if total_clusters > 0:
        print(f"    Same direction:  {same_dir_count:>6,} ({same_dir_count/total_clusters*100:.1f}%)")
        print(f"    Mixed direction: {mixed_dir_count:>6,} ({mixed_dir_count/total_clusters*100:.1f}%)")

    if cluster_pnls_aligned:
        a = np.array(cluster_pnls_aligned)
        print(f"\n    Aligned clusters avg PnL: {a.mean():+.2f}, "
              f"WR: {(a>0).mean()*100:.1f}%, total: {a.sum():+,.1f}")

    if cluster_pnls_mixed:
        m = np.array(cluster_pnls_mixed)
        print(f"    Mixed clusters avg PnL:   {m.mean():+.2f}, "
              f"WR: {(m>0).mean()*100:.1f}%, total: {m.sum():+,.1f}")

    # Currency exposure clustering
    print(f"\n  CURRENCY EXPOSURE CLUSTERING:")
    print(f"  (When 3+ entries fire simultaneously, how concentrated is the bet?)")

    big_clusters = []
    for ns, group in bar_groups:
        if len(group) < 3:
            continue

        currency_exposure = {}
        for _, row in group.iterrows():
            pair = row['pair']
            d = row['direction']
            base, quote = pair.split('_')
            currency_exposure[base] = currency_exposure.get(base, 0) + d
            currency_exposure[quote] = currency_exposure.get(quote, 0) - d

        max_exposure = max(abs(v) for v in currency_exposure.values())
        max_ccy = max(currency_exposure.keys(), key=lambda k: abs(currency_exposure[k]))
        cluster_pnl = group['pnl_pips'].sum()

        big_clusters.append({
            'n_entries': len(group),
            'max_exposure': max_exposure,
            'max_ccy': max_ccy,
            'exposure_dir': 'long' if currency_exposure[max_ccy] > 0 else 'short',
            'pnl': cluster_pnl,
            'time': pd.Timestamp(ns, unit='ns'),
        })

    if big_clusters:
        exposures = [c['max_exposure'] for c in big_clusters]
        print(f"\n    Clusters with 3+ concurrent entries: {len(big_clusters):,}")
        print(f"    Max single-currency exposure: {max(exposures)}")
        print(f"    Mean max exposure: {np.mean(exposures):.1f}")
        print(f"    P90 max exposure:  {np.percentile(exposures, 90):.0f}")

        concentrated = [c for c in big_clusters if c['max_exposure'] >= 5]
        print(f"\n    Concentrated clusters (exposure >= 5 on one ccy): {len(concentrated):,}")
        if concentrated:
            conc_pnls = [c['pnl'] for c in concentrated]
            print(f"    Their avg cluster PnL: {np.mean(conc_pnls):+.1f}")
            print(f"    Their total PnL:       {sum(conc_pnls):+,.1f}")
            print(f"    WR:                    {sum(1 for p in conc_pnls if p > 0)/len(conc_pnls)*100:.1f}%")

            ccy_counts = {}
            for c in concentrated:
                key = f"{c['exposure_dir']} {c['max_ccy']}"
                ccy_counts[key] = ccy_counts.get(key, 0) + 1
            print(f"\n    Most common concentrated bets:")
            for k, v in sorted(ccy_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"      {k}: {v} times")

        # Top 10 most concentrated moments
        top_conc = sorted(big_clusters, key=lambda c: -c['max_exposure'])[:10]
        print(f"\n    TOP 10 MOST CONCENTRATED MOMENTS:")
        print(f"    {'Entries':>8} {'Exposure':>10} {'Currency':>12} {'PnL':>10} {'Time':>22}")
        for c in top_conc:
            print(f"    {c['n_entries']:>8} {c['max_exposure']:>10} "
                  f"{c['exposure_dir']:>5} {c['max_ccy']:<4} {c['pnl']:>+10.1f} "
                  f"{c['time'].strftime('%Y-%m-%d %H:%M'):>22}")
    else:
        print(f"\n    No clusters with 3+ concurrent entries found")

    # JOINT P&L of concurrent entries — do they sink or swim together?
    print(f"\n  JOINT P&L CORRELATION:")
    print(f"  (When N entries open same bar, do they ALL win or ALL lose?)")

    for min_conc in [2, 3, 5, 8]:
        matching_ns = entries_per_bar[entries_per_bar >= min_conc].index
        if len(matching_ns) == 0:
            continue
        cluster_trades = df[df['entry_ns'].isin(matching_ns)]

        # For each cluster, what % of trades are winners?
        all_winner_pct = []
        for ns, group in cluster_trades.groupby('entry_ns'):
            wr = (group['pnl_pips'] > 0).mean()
            all_winner_pct.append(wr)

        aw = np.array(all_winner_pct)
        all_win = (aw == 1.0).mean() * 100
        all_lose = (aw == 0.0).mean() * 100
        mixed = ((aw > 0) & (aw < 1.0)).mean() * 100
        print(f"\n    Clusters with {min_conc}+ entries ({len(matching_ns):,} bars):")
        print(f"      All win:  {all_win:.1f}%")
        print(f"      All lose: {all_lose:.1f}%")
        print(f"      Mixed:    {mixed:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def discover_pairs(data_dir):
    pairs = {}
    for ext in ['*.parquet', '*.pkl']:
        for f in glob.glob(os.path.join(data_dir, ext)):
            base = os.path.basename(f).upper()
            for p in ALL_PAIRS:
                if (p in base or p.replace('_', '') in base) and p not in pairs:
                    pairs[p] = f
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Regime Acid Test v3 — Comprehensive Stress Test')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        for c in [r'C:\Forex_Projects\5_year_5s_data', r'C:\Forex_Projects\Forex_bot_original']:
            if os.path.isdir(c): data_dir = c; break
        if not data_dir: data_dir = os.getcwd()
    output_dir = args.output_dir or data_dir

    t_start = time_mod.time()

    print("=" * 115)
    print("  REGIME ACID TEST v3 — COMPREHENSIVE STRESS TEST")
    print("=" * 115)
    print(f"  Config: ET={ENTRY_THRESHOLD} XT={EXIT_THRESHOLD} TE={TIMED_EXIT*5/60:.0f}h")
    print(f"  Spread filters: {SPREAD_FILTERS}")
    print(f"  Slippage levels: {SLIPPAGE_LEVELS}")
    print(f"  Rollover exclusion: windows {sorted(ROLLOVER_WINDOWS)} (20:30-22:00 UTC)")
    print(f"  Data: {data_dir}")

    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files in {data_dir}")
        sys.exit(1)
    pair_names = sorted(pairs.keys())
    log.info(f"  Pairs: {len(pairs)} — {', '.join(pair_names)}")

    # ── PHASE 1: Load data, compute signals, generate trades ──
    all_trades = []

    for pname in pair_names:
        t0 = time_mod.time()
        df_5s = load_pair_data(pairs[pname])
        tfs = build_timeframes(df_5s)
        for tf in tfs:
            if len(tfs[tf]) >= 22:
                tfs[tf] = add_ema(tfs[tf])
        m5 = tfs['M5']
        if len(m5) < 100:
            log.warning(f"  {pname}: skip ({len(m5)} M5 bars)")
            continue

        bias = compute_mtf_bias(tfs, m5.index)
        state = regime_hysteresis(bias, ENTRY_THRESHOLD, EXIT_THRESHOLD)

        has_ba = 'bid_close' in m5.columns
        bc = m5['bid_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        ac = m5['ask_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        pip_m = 100.0 if 'JPY' in pname else 10000.0

        trades = generate_trades(state, m5.index, bc, ac, pname, pip_m)
        all_trades.extend(trades)
        log.info(f"  {pname}: {len(trades):,} raw trades ({time_mod.time()-t0:.1f}s)")

    del df_5s

    if not all_trades:
        log.error("No trades generated"); sys.exit(1)

    raw_df = pd.DataFrame(all_trades)
    log.info(f"\n  Raw trades (all pairs): {len(raw_df):,}")

    # ── PHASE 2: Position blocking ──
    blocked_df = apply_position_blocking(raw_df)
    n_blocked = len(raw_df) - len(blocked_df)
    log.info(f"  After position blocking: {len(blocked_df):,} ({n_blocked:,} blocked)")

    # ── PHASE 3: Spread distribution ──
    print(f"\n{'='*115}")
    print(f"  SPREAD DISTRIBUTION AT ENTRY")
    print(f"{'='*115}")

    sp = blocked_df['entry_spread_pips'].values
    print(f"\n  Mean:   {sp.mean():.2f} pips")
    print(f"  Median: {np.median(sp):.2f} pips")
    print(f"  P75:    {np.percentile(sp, 75):.2f} pips")
    print(f"  P90:    {np.percentile(sp, 90):.2f} pips")
    print(f"  P95:    {np.percentile(sp, 95):.2f} pips")
    print(f"  P99:    {np.percentile(sp, 99):.2f} pips")
    print(f"  Max:    {sp.max():.2f} pips")

    print(f"\n  {'Pair':>10} {'Trades':>8} {'Mean':>8} {'Median':>8} {'P75':>8} {'P90':>8} {'P99':>8} {'Max':>8}")
    for pair in sorted(blocked_df['pair'].unique()):
        g = blocked_df[blocked_df['pair'] == pair]
        s = g['entry_spread_pips'].values
        print(f"  {pair:>10} {len(g):>8,} {s.mean():>8.2f} {np.median(s):>8.2f} "
              f"{np.percentile(s,75):>8.2f} {np.percentile(s,90):>8.2f} "
              f"{np.percentile(s,99):>8.2f} {s.max():>8.1f}")

    # ── PHASE 4: Flat spread filter comparison ──
    results = []
    results.append(analyse_filtered(blocked_df, 'No filter (baseline)'))

    for max_spread in SPREAD_FILTERS:
        filtered = blocked_df[blocked_df['entry_spread_pips'] <= max_spread]
        results.append(analyse_filtered(filtered, f'Max spread <= {max_spread} pips'))

    no_rollover = blocked_df[~blocked_df['window'].isin(ROLLOVER_WINDOWS)]
    results.append(analyse_filtered(no_rollover, 'No rollover (skip 20:30-22:00)'))

    no_roll_spread5 = blocked_df[
        (~blocked_df['window'].isin(ROLLOVER_WINDOWS)) &
        (blocked_df['entry_spread_pips'] <= 5)
    ]
    results.append(analyse_filtered(no_roll_spread5, 'No rollover + spread <= 5'))

    no_roll_spread10 = blocked_df[
        (~blocked_df['window'].isin(ROLLOVER_WINDOWS)) &
        (blocked_df['entry_spread_pips'] <= 10)
    ]
    results.append(analyse_filtered(no_roll_spread10, 'No rollover + spread <= 10'))

    core = blocked_df[(blocked_df['window'] >= 14) & (blocked_df['window'] <= 39)]
    results.append(analyse_filtered(core, 'Core session only (07-20 UTC)'))

    print_comparison(results, "SPREAD FILTER COMPARISON — Position-blocked")

    # ── PHASE 5: Year-by-year for key configs ──
    configs = [
        ('No filter', blocked_df),
        ('Spread <= 5', blocked_df[blocked_df['entry_spread_pips'] <= 5]),
        ('No rollover', no_rollover),
        ('No rollover + spread <= 10', no_roll_spread10),
    ]

    for config_name, config_df in configs:
        print(f"\n{'='*115}")
        print(f"  YEARLY: {config_name}")
        print(f"{'='*115}")
        print(f"  {'Year':>6} {'Trades':>8} {'Total':>12} {'Avg':>8} {'WR':>6} {'PF':>6} {'DD':>8}")
        for yr, g in config_df.groupby('year'):
            pnl = g['pnl_pips'].values
            t = pnl.sum(); a = pnl.mean()
            w = (pnl > 0).mean() * 100
            gl = abs(pnl[pnl < 0].sum())
            p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
            cum = np.cumsum(pnl); dd = (np.maximum.accumulate(cum) - cum).max()
            marker = "+" if t > 0 else "-"
            print(f"  {yr:>6} {len(g):>8,} {t:>+12,.1f} {a:>+8.3f} {w:>5.1f}% {p:>6.2f} {dd:>8,.0f}  {marker}")

    # Losing months
    print(f"\n{'='*115}")
    print(f"  LOSING MONTHS COUNT")
    print(f"{'='*115}")
    for config_name, config_df in configs:
        cdf = config_df.copy()
        cdf['entry_dt'] = pd.to_datetime(cdf['entry_ns'], unit='ns')
        cdf['ym'] = cdf['entry_dt'].dt.to_period('M')
        monthly = cdf.groupby('ym')['pnl_pips'].sum()
        losing = (monthly < 0).sum()
        worst_month = monthly.min()
        best_month = monthly.max()
        print(f"  {config_name:<35} {losing} losing out of {len(monthly)} | "
              f"worst: {worst_month:+,.0f} | best: {best_month:+,.0f}")

    # ── PHASE 6: SLIPPAGE SIMULATION ──
    run_slippage_analysis(blocked_df)

    # ── PHASE 7: CONSECUTIVE LOSS STREAKS ──
    run_streak_analysis(blocked_df)

    # ── PHASE 8: DRAWDOWN DURATION ──
    run_drawdown_duration_analysis(blocked_df)

    # ── PHASE 9: PAIR-SPECIFIC ADAPTIVE SPREAD ──
    pair_p75 = run_adaptive_spread_analysis(blocked_df)

    # ── PHASE 10: CORRELATION / CLUSTERING ──
    run_correlation_analysis(blocked_df)

    # ── SAVE ──
    csv_path = os.path.join(output_dir, 'regime_acid_test_v3_spread.csv')
    blocked_df.to_csv(csv_path, index=False)
    log.info(f"\n  Trades CSV saved: {csv_path}")

    elapsed = time_mod.time() - t_start
    print(f"\n{'='*115}")
    print(f"  COMPLETE — {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  {len(blocked_df):,} trades analysed across {len(blocked_df['pair'].unique())} pairs")
    print(f"{'='*115}")


if __name__ == '__main__':
    main()
