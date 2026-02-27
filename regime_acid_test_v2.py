#!/usr/bin/env python3
"""
REGIME ACID TEST v2 — ENTRY ONLY, FIXED HOLD
==============================================
Tests whether the regime signal's value is purely as an ENTRY trigger.

Same as acid test v1 EXCEPT:
  - NO regime off exit
  - Hold for exactly N hours, then close
  - Tests multiple hold periods in one run: 1h, 2h, 4h, 6h, 8h, 12h, 16h, 24h, 48h

This answers: "Is the regime exit hurting us by cutting winners short?"
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
# CONFIG
# ══════════════════════════════════════════════════════════════════════

ENTRY_THRESHOLD = 0.4
EXIT_THRESHOLD  = 0.2    # still needed for state machine (entry detection)
ENTRY_CONFIRM   = 0

# ── FILTER TESTS (for v3 filter comparison) ──
SPREAD_FILTER_PIPS = 5.0           # reject entries with spread > this
TOXIC_HOURS_UTC    = [20, 21, 22]  # reject entries during these hours (rollover)

# Hold periods to test (in M5 bars)
HOLD_PERIODS = {
    '1h':  12,
    '2h':  24,
    '4h':  48,
    '6h':  72,
    '8h':  96,
    '12h': 144,
    '16h': 192,
    '24h': 288,
    '48h': 576,
}

MTF_WEIGHTS = {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}

ALL_PAIRS = ['EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
             'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
             'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
             'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD']

DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
MAX_HOLD_BARS = 576  # for array allocation

# ══════════════════════════════════════════════════════════════════════
# DATA / INDICATORS / REGIME (identical to acid test v1)
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
# TRADE GENERATION — FIXED HOLD, NO REGIME EXIT
# ══════════════════════════════════════════════════════════════════════

def generate_trades(state, m5_index, bid_c, ask_c, pair_name, pip_mult):
    """
    Generate trades: enter on regime ON, hold for EXACTLY hold_bars, exit.
    Regime OFF is IGNORED for exits. Pure timed exit only.
    
    Returns list of trade dicts with exit prices for ALL hold periods.
    """
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

            entry_price = ask_c[entry_bar] if d == 1 else bid_c[entry_bar]
            entry_ns = m5_ns[entry_bar]

            # Spread at entry bar (pips) — for filter testing
            spread_pips = (ask_c[entry_bar] - bid_c[entry_bar]) * pip_mult
            entry_hour = ts.hour

            # Also compute regime exit for comparison
            regime_exit_bar = None
            for j in range(entry_bar + 1, min(entry_bar + MAX_HOLD_BARS + 1, n)):
                if state[j] != d:
                    regime_exit_bar = j
                    break

            trade = {
                'pair': pair_name,
                'direction': d,
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'entry_ns': entry_ns,
                'year': ts.year,
                'dow': ts.dayofweek,
                'window': ts.hour * 2 + (1 if ts.minute >= 30 else 0),
                'entry_hour': entry_hour,
                'spread_pips': spread_pips,
            }

            # Compute exit price and PnL for each hold period
            for label, hold_bars in HOLD_PERIODS.items():
                exit_bar = min(entry_bar + hold_bars, n - 1)
                exit_price = bid_c[exit_bar] if d == 1 else ask_c[exit_bar]
                pnl = d * (exit_price - entry_price) * pip_mult
                trade[f'pnl_{label}'] = pnl
                trade[f'exit_price_{label}'] = exit_price

            # Regime exit comparison (if regime turned off before 48h)
            if regime_exit_bar and regime_exit_bar < n:
                re_price = bid_c[regime_exit_bar] if d == 1 else ask_c[regime_exit_bar]
                trade['pnl_regime_exit'] = d * (re_price - entry_price) * pip_mult
                trade['regime_exit_bar'] = regime_exit_bar
                trade['regime_hold_bars'] = regime_exit_bar - entry_bar
            else:
                # Regime never turned off within 48h — use 48h exit
                trade['pnl_regime_exit'] = trade['pnl_48h']
                trade['regime_exit_bar'] = entry_bar + MAX_HOLD_BARS
                trade['regime_hold_bars'] = MAX_HOLD_BARS

            # Original acid test exit (regime off OR 48h) for direct comparison
            if regime_exit_bar and regime_exit_bar - entry_bar <= MAX_HOLD_BARS:
                actual_exit = regime_exit_bar
            else:
                actual_exit = min(entry_bar + MAX_HOLD_BARS, n - 1)
            actual_ep = bid_c[actual_exit] if d == 1 else ask_c[actual_exit]
            trade['pnl_v1_regime_or_48h'] = d * (actual_ep - entry_price) * pip_mult

            trades.append(trade)

    return trades


# ══════════════════════════════════════════════════════════════════════
# POSITION-BLOCKED REPLAY
# ══════════════════════════════════════════════════════════════════════

def apply_blocking(df, pnl_col, hold_bars):
    """Per-pair position blocking for a specific hold period."""
    df = df.sort_values('entry_ns').reset_index(drop=True)
    last_exit = {}
    keep = []

    for idx, row in df.iterrows():
        pair = row['pair']
        exit_ns = row['entry_ns'] + hold_bars * 5 * 60 * 1_000_000  # microseconds
        if pair in last_exit and row['entry_ns'] < last_exit[pair]:
            continue
        keep.append(idx)
        last_exit[pair] = exit_ns

    return df.loc[keep].reset_index(drop=True)


def apply_blocking_regime(df):
    """Per-pair blocking using regime exit timing."""
    df = df.sort_values('entry_ns').reset_index(drop=True)
    last_exit = {}
    keep = []

    for idx, row in df.iterrows():
        pair = row['pair']
        exit_ns = row['entry_ns'] + int(row['regime_hold_bars']) * 5 * 60 * 1_000_000
        if pair in last_exit and row['entry_ns'] < last_exit[pair]:
            continue
        keep.append(idx)
        last_exit[pair] = exit_ns

    return df.loc[keep].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════

def summarise(df, pnl_col, label):
    """Compute summary stats for a given pnl column."""
    pnl = df[pnl_col].values
    n = len(pnl)
    total = pnl.sum()
    avg = pnl.mean()
    wr = (pnl > 0).mean() * 100
    gl = abs(pnl[pnl < 0].sum())
    pf = pnl[pnl > 0].sum() / gl if gl > 0 else 999
    cum = np.cumsum(pnl)
    dd = (np.maximum.accumulate(cum) - cum).max()
    avg_win = pnl[pnl > 0].mean() if (pnl > 0).any() else 0
    avg_loss = pnl[pnl < 0].mean() if (pnl < 0).any() else 0

    return {
        'label': label,
        'trades': n,
        'total_pips': total,
        'avg_pips': avg,
        'win_rate': wr,
        'profit_factor': pf,
        'max_dd': dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }


def print_comparison(results):
    w = 100
    print("\n" + "=" * w)
    print("  REGIME EXIT vs FIXED HOLD — HEAD-TO-HEAD COMPARISON")
    print(f"  Entry: ET={ENTRY_THRESHOLD} XT={EXIT_THRESHOLD} EC={ENTRY_CONFIRM*5}min")
    print(f"  All results position-blocked per pair")
    print("=" * w)

    print(f"\n  {'Exit Strategy':<25} {'Trades':>8} {'Total Pips':>14} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>10} {'Avg W':>8} {'Avg L':>8}")
    print("  " + "-" * 95)

    for r in results:
        print(f"  {r['label']:<25} {r['trades']:>8,} {r['total_pips']:>+14,.1f} "
              f"{r['avg_pips']:>+8.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['max_dd']:>10,.0f} {r['avg_win']:>+8.2f} {r['avg_loss']:>+8.2f}")

    # Find best
    best = max(results, key=lambda r: r['total_pips'])
    print(f"\n  ★ BEST: {best['label']} at {best['total_pips']:+,.1f} total pips")

    # Compare regime exit vs best fixed
    regime = next((r for r in results if 'Regime' in r['label'] and 'v1' not in r['label'].lower()), None)
    v1 = next((r for r in results if 'v1' in r['label'].lower()), None)

    if regime and best['label'] != regime['label']:
        diff = best['total_pips'] - regime['total_pips']
        pct = diff / abs(regime['total_pips']) * 100 if regime['total_pips'] != 0 else 0
        print(f"    vs regime-only exit: {diff:+,.1f} pips ({pct:+.1f}%)")

    if v1 and best['label'] != v1['label']:
        diff = best['total_pips'] - v1['total_pips']
        pct = diff / abs(v1['total_pips']) * 100 if v1['total_pips'] != 0 else 0
        print(f"    vs v1 (regime OR 48h): {diff:+,.1f} pips ({pct:+.1f}%)")


def print_yearly(df, pnl_col, label):
    print(f"\n── {label} — BY YEAR ──")
    print(f"  {'Year':>6} {'Trades':>8} {'Total':>12} {'Avg':>8} {'WR':>6} {'PF':>6}")
    for yr, g in df.groupby('year'):
        pnl = g[pnl_col].values
        t = pnl.sum()
        a = pnl.mean()
        w = (pnl > 0).mean() * 100
        gl = abs(pnl[pnl < 0].sum())
        p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
        marker = "✓" if t > 0 else "✗"
        print(f"  {yr:>6} {len(g):>8,} {t:>+12,.1f} {a:>+8.3f} {w:>5.1f}% {p:>6.2f}  {marker}")


def print_pair_comparison(df, col1, col2, label1, label2):
    """Show per-pair which exit strategy wins."""
    print(f"\n── PER PAIR: {label1} vs {label2} ──")
    print(f"  {'Pair':>10} {label1:>14} {label2:>14} {'Diff':>12} {'Winner':>10}")
    for pair in sorted(df['pair'].unique()):
        g = df[df['pair'] == pair]
        t1 = g[col1].sum()
        t2 = g[col2].sum()
        diff = t2 - t1
        winner = label2 if diff > 0 else label1
        print(f"  {pair:>10} {t1:>+14,.1f} {t2:>+14,.1f} {diff:>+12,.1f} {winner:>10}")


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
    parser = argparse.ArgumentParser(description='Regime Acid Test v2 — Fixed Hold')
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

    print("=" * 100)
    print("  REGIME ACID TEST v2 — ENTRY ONLY, FIXED HOLD (NO REGIME EXIT)")
    print("=" * 100)
    print(f"  ET={ENTRY_THRESHOLD}  XT={EXIT_THRESHOLD}  EC={ENTRY_CONFIRM*5}min")
    print(f"  Hold periods: {', '.join(HOLD_PERIODS.keys())}")
    print(f"  Regime exit: DISABLED for timed holds, tracked for comparison")
    print(f"  Filter test: spread>{SPREAD_FILTER_PIPS}pip reject, toxic hours {TOXIC_HOURS_UTC} reject")

    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files in {data_dir}"); sys.exit(1)
    pair_names = sorted(pairs.keys())
    log.info(f"  Pairs: {len(pairs)} — {', '.join(pair_names)}")

    # ── Generate trades ──
    all_trades = []
    for pname in pair_names:
        t0 = time_mod.time()
        df_5s = load_pair_data(pairs[pname])
        tfs = build_timeframes(df_5s)
        for tf in tfs:
            if len(tfs[tf]) >= 22: tfs[tf] = add_ema(tfs[tf])
        m5 = tfs['M5']
        if len(m5) < 100:
            log.warning(f"  {pname}: skip ({len(m5)} M5 bars)"); continue
        bias = compute_mtf_bias(tfs, m5.index)
        state = regime_hysteresis(bias, ENTRY_THRESHOLD, EXIT_THRESHOLD)
        has_ba = 'bid_close' in m5.columns
        bc = m5['bid_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        ac = m5['ask_close'].values.astype(np.float64) if has_ba else m5['close'].values.astype(np.float64)
        pm = 100.0 if 'JPY' in pname else 10000.0
        trades = generate_trades(state, m5.index, bc, ac, pname, pm)
        all_trades.extend(trades)
        log.info(f"  {pname}: {len(trades):,} trades ({time_mod.time()-t0:.1f}s)")
    del df_5s

    if not all_trades:
        log.error("No trades"); sys.exit(1)

    raw_df = pd.DataFrame(all_trades)
    log.info(f"\n  Raw trades: {len(raw_df):,}")

    # ── Apply position blocking and compute stats for each hold period ──
    results = []

    # 1. Regime-only exit (for comparison)
    regime_df = apply_blocking_regime(raw_df.copy())
    results.append(summarise(regime_df, 'pnl_regime_exit', 'Regime exit only'))

    # 2. v1 acid test (regime OR 48h — what we already ran)
    # Need to block using regime exit timing since that's what v1 did
    results.append(summarise(regime_df, 'pnl_v1_regime_or_48h', 'v1 (regime OR 48h)'))

    # 3. Fixed hold periods
    for label, hold_bars in HOLD_PERIODS.items():
        blocked = apply_blocking(raw_df.copy(), f'pnl_{label}', hold_bars)
        results.append(summarise(blocked, f'pnl_{label}', f'Fixed {label}'))

    # ── Print comparison ──
    print_comparison(results)

    # ── Yearly breakdown for top 3 ──
    results_sorted = sorted(results, key=lambda r: r['total_pips'], reverse=True)
    for r in results_sorted[:3]:
        label = r['label']
        if 'Fixed' in label:
            hold_label = label.replace('Fixed ', '')
            hold_bars = HOLD_PERIODS[hold_label]
            blocked = apply_blocking(raw_df.copy(), f'pnl_{hold_label}', hold_bars)
            print_yearly(blocked, f'pnl_{hold_label}', label)
        elif 'regime OR' in label.lower() or 'v1' in label.lower():
            print_yearly(regime_df, 'pnl_v1_regime_or_48h', label)
        else:
            print_yearly(regime_df, 'pnl_regime_exit', label)

    # ── Per-pair comparison: best fixed vs regime ──
    best_fixed = next((r for r in results_sorted if 'Fixed' in r['label']), None)
    if best_fixed:
        hold_label = best_fixed['label'].replace('Fixed ', '')
        hold_bars = HOLD_PERIODS[hold_label]
        blocked = apply_blocking(raw_df.copy(), f'pnl_{hold_label}', hold_bars)
        # Merge regime pnl into blocked df for comparison
        blocked_with_regime = blocked.copy()
        print_pair_comparison(blocked_with_regime, 'pnl_regime_exit',
                              f'pnl_{hold_label}', 'Regime exit', best_fixed['label'])

    # ══════════════════════════════════════════════════════════════════
    # FILTER COMPARISON — spread filter + toxic hour filter
    # Tests what the live bot's filters would have done historically
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "=" * 100)
    print("  FILTER IMPACT ANALYSIS — regime exit (v1) strategy")
    print(f"  Spread filter: reject entries > {SPREAD_FILTER_PIPS} pips")
    print(f"  Toxic hours:   reject entries during {TOXIC_HOURS_UTC} UTC")
    print("=" * 100)

    # Spread stats
    spreads = raw_df['spread_pips']
    print(f"\n  Spread stats across all {len(raw_df):,} raw entries:")
    print(f"    Mean: {spreads.mean():.2f} pips  |  Median: {spreads.median():.2f}  |  "
          f"95th: {spreads.quantile(0.95):.2f}  |  99th: {spreads.quantile(0.99):.2f}  |  Max: {spreads.max():.1f}")
    print(f"    Entries rejected by {SPREAD_FILTER_PIPS}-pip filter: "
          f"{(spreads > SPREAD_FILTER_PIPS).sum():,} ({(spreads > SPREAD_FILTER_PIPS).mean()*100:.1f}%)")
    print(f"    Entries rejected by toxic hours: "
          f"{raw_df['entry_hour'].isin(TOXIC_HOURS_UTC).sum():,} ({raw_df['entry_hour'].isin(TOXIC_HOURS_UTC).mean()*100:.1f}%)")

    # Build filtered DataFrames
    no_filter    = raw_df.copy()
    spread_only  = raw_df[raw_df['spread_pips'] <= SPREAD_FILTER_PIPS].copy()
    time_only    = raw_df[~raw_df['entry_hour'].isin(TOXIC_HOURS_UTC)].copy()
    both_filters = raw_df[(raw_df['spread_pips'] <= SPREAD_FILTER_PIPS) &
                          (~raw_df['entry_hour'].isin(TOXIC_HOURS_UTC))].copy()

    filter_sets = [
        ('No filter (baseline)',        no_filter),
        (f'Spread <= {SPREAD_FILTER_PIPS} pips',   spread_only),
        ('No toxic hours (20-22)',      time_only),
        ('Spread + No toxic hours',     both_filters),
    ]

    filter_results = []
    for label, fdf in filter_sets:
        blocked = apply_blocking_regime(fdf)
        r = summarise(blocked, 'pnl_v1_regime_or_48h', label)
        filter_results.append(r)

    print(f"\n  {'Filter':<30} {'Trades':>8} {'Total Pips':>14} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>10}")
    print("  " + "-" * 85)
    for r in filter_results:
        print(f"  {r['label']:<30} {r['trades']:>8,} {r['total_pips']:>+14,.1f} "
              f"{r['avg_pips']:>+8.3f} {r['win_rate']:>5.1f}% {r['profit_factor']:>6.2f} "
              f"{r['max_dd']:>10,.0f}")

    # Delta from baseline
    base_pips = filter_results[0]['total_pips']
    print(f"\n  Delta from baseline:")
    for r in filter_results[1:]:
        diff = r['total_pips'] - base_pips
        pct = diff / abs(base_pips) * 100 if base_pips != 0 else 0
        print(f"    {r['label']:<30} {diff:>+12,.1f} pips ({pct:>+5.1f}%)")

    # Yearly breakdown for best filter combo
    best_filter = max(filter_results, key=lambda r: r['total_pips'])
    if best_filter['label'] != filter_results[0]['label']:
        print(f"\n  ★ BEST FILTER: {best_filter['label']}")
        # Find matching filtered df
        for label, fdf in filter_sets:
            if label == best_filter['label']:
                blocked = apply_blocking_regime(fdf)
                print_yearly(blocked, 'pnl_v1_regime_or_48h',
                             f"{best_filter['label']} — yearly")
                break

    # Toxic hour detail
    print(f"\n  ── TOXIC HOUR DETAIL (entry hour → regime exit PnL) ──")
    print(f"  {'Hour':>6} {'Trades':>8} {'Total':>12} {'Avg':>8} {'WR':>6} {'Avg Spread':>10}")
    for h in range(24):
        g = raw_df[raw_df['entry_hour'] == h]
        if len(g) == 0: continue
        pnl = g['pnl_v1_regime_or_48h']
        marker = " ← TOXIC" if h in TOXIC_HOURS_UTC else ""
        marker2 = " ← WORST" if pnl.mean() < -3 else marker
        print(f"  {h:>6} {len(g):>8,} {pnl.sum():>+12,.1f} {pnl.mean():>+8.3f} "
              f"{(pnl>0).mean()*100:>5.1f}% {g['spread_pips'].mean():>10.2f}{marker2}")

    # ── Save ──
    csv_path = os.path.join(output_dir, 'regime_acid_test_v2_holdcomp.csv')
    raw_df.to_csv(csv_path, index=False)
    log.info(f"\n  Saved: {csv_path}")

    elapsed = time_mod.time() - t_start
    log.info(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print("\n" + "=" * 100)


if __name__ == '__main__':
    main()
