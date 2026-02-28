#!/usr/bin/env python3
"""
COINTEGRATION MEAN-REVERSION BACKTEST
======================================
Strategy: AUD/NZD and EUR/GBP mean-revert because their economies are
structurally linked.  When price deviates from its rolling mean by Z
standard deviations, fade the move and wait for reversion.

Anti-cheat measures:
  1. Rolling lookback uses ONLY past bars (no future data)
  2. Entry on NEXT bar after signal (no same-bar execution)
  3. Bid/ask pricing: buy at ask, sell at bid (realistic fills)
  4. Spread filter at entry
  5. Slippage simulation
  6. Year-by-year breakdown for consistency
  7. CONTROL PAIRS: EUR_USD, GBP_USD tested alongside — if they show
     similar edge, the strategy is NOT capturing mean-reversion
  8. Reports ALL parameter combos — no cherry-picking

Run:
  python coint_backtest.py --data-dir /path/to/5s/parquets
"""

import os, sys, glob, argparse, time as time_mod, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════

# Target pairs — structurally linked, expected to mean-revert
TARGET_PAIRS = ['AUD_NZD', 'EUR_GBP']

# Control pairs — NOT expected to mean-revert.  If these show similar
# edge, the strategy is bogus.
CONTROL_PAIRS = ['EUR_USD', 'GBP_USD']

ALL_TEST_PAIRS = TARGET_PAIRS + CONTROL_PAIRS

# Pip multipliers
PIP_MULT = {
    'AUD_NZD': 10_000.0,
    'EUR_GBP': 10_000.0,
    'EUR_USD': 10_000.0,
    'GBP_USD': 10_000.0,
}

# ── Parameter grid ──
# Lookback windows for rolling mean/std (in M5 bars)
LOOKBACK_WINDOWS = {
    '48h':   576,
    '1wk':  2016,
    '2wk':  4032,
}

# Z-score thresholds for entry
ENTRY_Z_THRESHOLDS = [1.5, 2.0, 2.5, 3.0]

# Exit modes
EXIT_MODES = {
    'z_cross_0':   0.0,    # exit when z-score crosses zero
    'z_cross_0.5': 0.5,    # exit when |z| < 0.5
}

# Max hold (safety cap) in M5 bars
MAX_HOLD_BARS = {
    '24h': 288,
    '48h': 576,
}

# Spread filter (pips)
MAX_SPREAD_PIPS = 5.0

# Rollover exclusion windows (20:30–22:00 UTC → windows 41,42,43)
ROLLOVER_WINDOWS = {41, 42, 43}

# Slippage levels for robustness test
SLIPPAGE_LEVELS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
RNG_SEED = 42


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING — identical to existing infrastructure
# ══════════════════════════════════════════════════════════════════════

def load_pair_data(filepath):
    """Load 5-second parquet data."""
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


def resample_to_m5(df_5s):
    """Resample 5-second data to M5 bars."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df_5s.columns:
        agg['volume'] = 'sum'
    has_ba = all(c in df_5s.columns for c in
                 ['bid_high', 'bid_low', 'bid_close',
                  'ask_high', 'ask_low', 'ask_close'])
    if has_ba:
        for c, f in {'bid_open': 'first', 'bid_high': 'max',
                      'bid_low': 'min', 'bid_close': 'last',
                      'ask_open': 'first', 'ask_high': 'max',
                      'ask_low': 'min', 'ask_close': 'last'}.items():
            if c in df_5s.columns:
                agg[c] = f
    m5 = df_5s.resample('5min').agg(agg).dropna(subset=['close'])
    return m5


# ══════════════════════════════════════════════════════════════════════
# Z-SCORE COMPUTATION — the core signal
# ══════════════════════════════════════════════════════════════════════

def compute_rolling_zscore(close, lookback):
    """
    Compute rolling z-score using ONLY past data.

    z[i] = (close[i] - mean(close[i-lookback:i])) / std(close[i-lookback:i])

    First `lookback` bars have no signal (NaN).
    Uses ddof=1 for unbiased std estimate.
    """
    n = len(close)
    z = np.full(n, np.nan, dtype=np.float64)

    # Use pandas rolling for correctness and speed
    s = pd.Series(close)
    roll_mean = s.rolling(window=lookback, min_periods=lookback).mean().values
    roll_std = s.rolling(window=lookback, min_periods=lookback).std(ddof=1).values

    # Only compute where we have full lookback AND non-zero std
    valid = (~np.isnan(roll_mean)) & (roll_std > 1e-12)
    z[valid] = (close[valid] - roll_mean[valid]) / roll_std[valid]

    return z


# ══════════════════════════════════════════════════════════════════════
# TRADE GENERATION — strict anti-cheat
# ══════════════════════════════════════════════════════════════════════

def generate_trades(zscore, m5_index, bid_c, ask_c, pair_name, pip_mult,
                    entry_z, exit_z, max_hold):
    """
    Generate mean-reversion trades.

    Entry rules:
      - z > +entry_z on bar i  →  SHORT on bar i+1 (enter at bid)
      - z < -entry_z on bar i  →  LONG  on bar i+1 (enter at ask)
      - Signal bar i, execution bar i+1 (no same-bar cheating)

    Exit rules:
      - z crosses exit_z threshold toward zero  →  exit on NEXT bar
      - OR max hold bars reached  →  forced exit
      - Whichever comes first

    Pricing:
      - LONG entry: ask price  |  LONG exit: bid price
      - SHORT entry: bid price |  SHORT exit: ask price
    """
    n = len(zscore)
    m5_ns = m5_index.asi8
    trades = []

    i = 0
    while i < n - 2:  # need at least signal bar + entry bar + 1 exit bar
        z_signal = zscore[i]

        # Skip if no valid z-score
        if np.isnan(z_signal):
            i += 1
            continue

        # Check entry conditions on signal bar
        direction = 0
        if z_signal > entry_z:
            direction = -1   # SHORT — price is high, expect reversion down
        elif z_signal < -entry_z:
            direction = 1    # LONG  — price is low, expect reversion up

        if direction == 0:
            i += 1
            continue

        # Execute on NEXT bar (i+1)
        entry_bar = i + 1
        if entry_bar >= n:
            break

        # Weekend filter
        ts = m5_index[entry_bar]
        if ts.dayofweek >= 5:
            i += 1
            continue

        # Rollover filter
        window = ts.hour * 2 + (1 if ts.minute >= 30 else 0)
        if window in ROLLOVER_WINDOWS:
            i += 1
            continue

        # Spread filter
        entry_spread = (ask_c[entry_bar] - bid_c[entry_bar]) * pip_mult
        if entry_spread > MAX_SPREAD_PIPS:
            i += 1
            continue

        # Entry price — LONG buys at ask, SHORT sells at bid
        entry_price = ask_c[entry_bar] if direction == 1 else bid_c[entry_bar]
        entry_ns = m5_ns[entry_bar]

        # Find exit
        exit_bar = min(entry_bar + max_hold, n - 1)
        exit_reason = 'max_hold'

        for j in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
            z_j = zscore[j]
            if np.isnan(z_j):
                continue

            # Check if z-score has reverted past exit threshold
            if direction == 1 and z_j >= -exit_z:
                # Was long (z was very negative), now z >= -exit_z → reverted
                exit_bar = min(j + 1, n - 1)  # exit on NEXT bar
                exit_reason = 'z_revert'
                break
            elif direction == -1 and z_j <= exit_z:
                # Was short (z was very positive), now z <= exit_z → reverted
                exit_bar = min(j + 1, n - 1)
                exit_reason = 'z_revert'
                break

        # Exit price — LONG sells at bid, SHORT buys at ask
        exit_price = bid_c[exit_bar] if direction == 1 else ask_c[exit_bar]
        exit_ns = m5_ns[exit_bar]

        # PnL in pips
        pnl = direction * (exit_price - entry_price) * pip_mult
        hold_bars = exit_bar - entry_bar

        trades.append({
            'pair': pair_name,
            'direction': direction,
            'entry_bar': entry_bar,
            'exit_bar': exit_bar,
            'entry_ns': entry_ns,
            'exit_ns': exit_ns,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_spread_pips': entry_spread,
            'entry_z': z_signal,
            'pnl_pips': pnl,
            'hold_bars': hold_bars,
            'hold_hours': hold_bars * 5 / 60,
            'exit_reason': exit_reason,
            'year': ts.year,
            'dow': ts.dayofweek,
            'window': window,
        })

        # Skip forward past this trade's exit (position blocking inline)
        i = exit_bar
        continue

    return trades


# ══════════════════════════════════════════════════════════════════════
# POSITION BLOCKING — per-pair, no overlapping trades
# ══════════════════════════════════════════════════════════════════════

def apply_position_blocking(trades_df):
    """Block overlapping trades per pair."""
    if len(trades_df) == 0:
        return trades_df
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

def compute_stats(pnl_arr, label=''):
    """Compute trading statistics from PnL array."""
    n = len(pnl_arr)
    if n == 0:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0,
                'wr': 0, 'pf': 0, 'max_dd': 0, 'sharpe': 0}
    total = pnl_arr.sum()
    avg = pnl_arr.mean()
    wr = (pnl_arr > 0).mean() * 100
    gross_loss = abs(pnl_arr[pnl_arr < 0].sum())
    pf = pnl_arr[pnl_arr > 0].sum() / gross_loss if gross_loss > 0 else 999
    cum = np.cumsum(pnl_arr)
    max_dd = (np.maximum.accumulate(cum) - cum).max()

    # Sharpe-like ratio (annualized, assuming M5 bars)
    bars_per_year = 252 * 24 * 12  # ~72,576
    if pnl_arr.std() > 0 and n > 1:
        sharpe = (avg / pnl_arr.std()) * np.sqrt(min(bars_per_year, n))
    else:
        sharpe = 0

    return {
        'label': label, 'n': n, 'total': total, 'avg': avg,
        'wr': wr, 'pf': pf, 'max_dd': max_dd, 'sharpe': sharpe,
    }


def print_header(title, width=120):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_stats_table(results, title="RESULTS"):
    print_header(title)
    print(f"\n  {'Config':<55} {'Trades':>7} {'Total Pips':>12} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>9} {'Sharpe':>7}")
    print("  " + "-" * 112)
    for r in results:
        if r['n'] == 0:
            print(f"  {r['label']:<55} {'0':>7}")
            continue
        print(f"  {r['label']:<55} {r['n']:>7,} {r['total']:>+12,.1f} "
              f"{r['avg']:>+8.3f} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{r['max_dd']:>9,.0f} {r['sharpe']:>7.2f}")


def print_yearly(trades_df, label):
    """Year-by-year breakdown."""
    print(f"\n  {label} — BY YEAR")
    print(f"  {'Year':>6} {'Trades':>8} {'Total':>12} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>8}")
    for yr in sorted(trades_df['year'].unique()):
        g = trades_df[trades_df['year'] == yr]
        pnl = g['pnl_pips'].values
        t = pnl.sum()
        a = pnl.mean()
        w = (pnl > 0).mean() * 100
        gl = abs(pnl[pnl < 0].sum())
        p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
        cum = np.cumsum(pnl)
        dd = (np.maximum.accumulate(cum) - cum).max() if len(cum) > 0 else 0
        marker = "+" if t > 0 else "-"
        print(f"  {yr:>6} {len(g):>8,} {t:>+12,.1f} {a:>+8.3f} "
              f"{w:>5.1f}% {p:>6.2f} {dd:>8,.0f}  {marker}")


# ══════════════════════════════════════════════════════════════════════
# SLIPPAGE SIMULATION
# ══════════════════════════════════════════════════════════════════════

def run_slippage_test(trades_df, label):
    """Test robustness under adverse slippage."""
    if len(trades_df) == 0:
        return

    base_pnl = trades_df['pnl_pips'].values
    base_total = base_pnl.sum()

    print(f"\n  SLIPPAGE ROBUSTNESS — {label}")
    print(f"  {'Slip (pips)':>12} {'Avg Total':>12} {'Avg/Trade':>10} "
          f"{'Delta %':>8} {'WR':>6} {'PF':>6}")
    print("  " + "-" * 65)

    rng = np.random.RandomState(RNG_SEED)
    n_runs = 10

    for slip in SLIPPAGE_LEVELS:
        if slip == 0:
            print(f"  {'0 (base)':>12} {base_total:>+12,.1f} "
                  f"{base_pnl.mean():>+10.3f} {'—':>8} "
                  f"{(base_pnl>0).mean()*100:>5.1f}% "
                  f"{base_pnl[base_pnl>0].sum()/max(abs(base_pnl[base_pnl<0].sum()),1e-9):>6.2f}")
            continue

        run_totals = []
        run_wrs = []
        run_pfs = []
        for _ in range(n_runs):
            adverse = rng.uniform(0, slip, size=len(base_pnl))
            adj_pnl = base_pnl - adverse  # slippage always hurts
            run_totals.append(adj_pnl.sum())
            run_wrs.append((adj_pnl > 0).mean() * 100)
            gl = abs(adj_pnl[adj_pnl < 0].sum())
            run_pfs.append(adj_pnl[adj_pnl > 0].sum() / gl if gl > 0 else 999)

        avg_total = np.mean(run_totals)
        avg_per = avg_total / len(base_pnl)
        delta_pct = (avg_total - base_total) / abs(base_total) * 100 if base_total != 0 else 0
        avg_wr = np.mean(run_wrs)
        avg_pf = np.mean(run_pfs)

        still_pos = "OK" if avg_total > 0 else "NEGATIVE"
        print(f"  {slip:>12.1f} {avg_total:>+12,.1f} {avg_per:>+10.3f} "
              f"{delta_pct:>+7.1f}% {avg_wr:>5.1f}% {avg_pf:>6.2f}  {still_pos}")


# ══════════════════════════════════════════════════════════════════════
# LEGITIMACY TESTS — is the edge real?
# ══════════════════════════════════════════════════════════════════════

def run_legitimacy_tests(target_results, control_results):
    """
    Compare target pairs vs control pairs.
    If control pairs show similar edge, the strategy is NOT capturing
    mean-reversion — it's just curve-fitting or exploiting something else.
    """
    print_header("LEGITIMACY CHECK — TARGET vs CONTROL PAIRS")

    print(f"\n  TARGET pairs (should show edge — structurally mean-reverting):")
    for r in target_results:
        if r['n'] > 0:
            print(f"    {r['label']:<40} {r['n']:>6,} trades  "
                  f"{r['total']:>+10,.1f} pips  avg {r['avg']:>+.3f}  "
                  f"WR {r['wr']:.1f}%  PF {r['pf']:.2f}")
        else:
            print(f"    {r['label']:<40} no trades")

    print(f"\n  CONTROL pairs (should show NO edge — not mean-reverting):")
    for r in control_results:
        if r['n'] > 0:
            print(f"    {r['label']:<40} {r['n']:>6,} trades  "
                  f"{r['total']:>+10,.1f} pips  avg {r['avg']:>+.3f}  "
                  f"WR {r['wr']:.1f}%  PF {r['pf']:.2f}")
        else:
            print(f"    {r['label']:<40} no trades")

    # Verdict
    target_avg = np.mean([r['avg'] for r in target_results if r['n'] > 0]) if target_results else 0
    control_avg = np.mean([r['avg'] for r in control_results if r['n'] > 0]) if control_results else 0

    print(f"\n  VERDICT:")
    if target_avg > 0 and control_avg <= 0:
        print(f"    Target avg/trade: {target_avg:+.3f} pips  |  Control avg/trade: {control_avg:+.3f} pips")
        print(f"    PASS — Edge appears specific to mean-reverting pairs")
    elif target_avg > 0 and control_avg > 0:
        ratio = target_avg / control_avg if control_avg > 0 else 999
        print(f"    Target avg/trade: {target_avg:+.3f} pips  |  Control avg/trade: {control_avg:+.3f} pips")
        if ratio > 2:
            print(f"    MARGINAL — Target outperforms control {ratio:.1f}x, but control also profitable")
        else:
            print(f"    FAIL — Control pairs show similar edge. Strategy is NOT mean-reversion specific")
    else:
        print(f"    Target avg/trade: {target_avg:+.3f} pips  |  Control avg/trade: {control_avg:+.3f} pips")
        print(f"    FAIL — No edge detected on target pairs")


# ══════════════════════════════════════════════════════════════════════
# CONSECUTIVE LOSS STREAK & DRAWDOWN ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def run_streak_analysis(trades_df, label):
    """Analyse consecutive loss streaks."""
    if len(trades_df) == 0:
        return
    pnl = trades_df['pnl_pips'].values
    wins = pnl > 0

    # Consecutive losses
    max_loss_streak = 0
    cur_streak = 0
    for w in wins:
        if not w:
            cur_streak += 1
            max_loss_streak = max(max_loss_streak, cur_streak)
        else:
            cur_streak = 0

    # Drawdown duration (in trades)
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    in_dd = cum < peak
    max_dd_duration = 0
    cur_dd = 0
    for d in in_dd:
        if d:
            cur_dd += 1
            max_dd_duration = max(max_dd_duration, cur_dd)
        else:
            cur_dd = 0

    print(f"\n  STREAK ANALYSIS — {label}")
    print(f"    Max consecutive losses: {max_loss_streak}")
    print(f"    Max drawdown duration:  {max_dd_duration} trades")
    print(f"    Total trades:           {len(pnl):,}")
    print(f"    Max drawdown (pips):    {(peak - cum).max():,.1f}")


# ══════════════════════════════════════════════════════════════════════
# DATA DISCOVERY
# ══════════════════════════════════════════════════════════════════════

def discover_pairs(data_dir):
    """Find parquet files matching our test pairs."""
    pairs = {}
    for ext in ['*.parquet', '*.pkl']:
        for f in glob.glob(os.path.join(data_dir, ext)):
            base = os.path.basename(f).upper()
            for p in ALL_TEST_PAIRS:
                if (p in base or p.replace('_', '') in base) and p not in pairs:
                    pairs[p] = f
    return pairs


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Cointegration Mean-Reversion Backtest')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing parquet files')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output files')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        for c in [r'C:\Forex_Projects\5_year_5s_data',
                  r'C:\Forex_Projects\Forex_bot_original']:
            if os.path.isdir(c):
                data_dir = c
                break
        if not data_dir:
            data_dir = os.getcwd()
    output_dir = args.output_dir or data_dir

    t_start = time_mod.time()

    print_header("COINTEGRATION MEAN-REVERSION BACKTEST")
    print(f"  Target pairs (mean-reverting): {TARGET_PAIRS}")
    print(f"  Control pairs (trending):      {CONTROL_PAIRS}")
    print(f"  Lookback windows: {list(LOOKBACK_WINDOWS.keys())}")
    print(f"  Entry Z thresholds: {ENTRY_Z_THRESHOLDS}")
    print(f"  Exit modes: {list(EXIT_MODES.keys())}")
    print(f"  Max hold: {list(MAX_HOLD_BARS.keys())}")
    print(f"  Spread filter: {MAX_SPREAD_PIPS} pips")
    print(f"  Rollover exclusion: windows {sorted(ROLLOVER_WINDOWS)}")
    print(f"  Data: {data_dir}")

    # ── Discover pairs ──
    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files found in {data_dir}")
        sys.exit(1)

    found_target = [p for p in TARGET_PAIRS if p in pairs]
    found_control = [p for p in CONTROL_PAIRS if p in pairs]
    log.info(f"  Target pairs found: {found_target}")
    log.info(f"  Control pairs found: {found_control}")

    if not found_target:
        log.error("No target pairs found. Need AUD_NZD or EUR_GBP.")
        sys.exit(1)

    # ── Load and resample all pairs ──
    pair_m5 = {}
    for pname, fpath in pairs.items():
        t0 = time_mod.time()
        df_5s = load_pair_data(fpath)
        m5 = resample_to_m5(df_5s)
        pair_m5[pname] = m5
        log.info(f"  {pname}: {len(df_5s):,} 5s bars → {len(m5):,} M5 bars "
                 f"({time_mod.time()-t0:.1f}s)")
        del df_5s  # free memory

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: GRID SEARCH — all parameter combinations
    # ══════════════════════════════════════════════════════════════════

    print_header("PHASE 1 — PARAMETER GRID SEARCH")

    all_results = []      # every combo's stats
    all_trades_by_key = {}  # key → trades DataFrame
    combo_count = 0
    total_combos = (len(LOOKBACK_WINDOWS) * len(ENTRY_Z_THRESHOLDS) *
                    len(EXIT_MODES) * len(MAX_HOLD_BARS) * len(pairs))

    print(f"  Total combos to test: {total_combos}")

    for lb_name, lb_bars in LOOKBACK_WINDOWS.items():
        for ez in ENTRY_Z_THRESHOLDS:
            for exit_name, exit_z in EXIT_MODES.items():
                for mh_name, mh_bars in MAX_HOLD_BARS.items():
                    for pname in sorted(pairs.keys()):
                        combo_count += 1
                        m5 = pair_m5[pname]
                        pip_m = PIP_MULT.get(pname, 10000.0)

                        # Get bid/ask or fall back to close
                        has_ba = 'bid_close' in m5.columns
                        if has_ba:
                            bid_c = m5['bid_close'].values.astype(np.float64)
                            ask_c = m5['ask_close'].values.astype(np.float64)
                        else:
                            bid_c = m5['close'].values.astype(np.float64)
                            ask_c = m5['close'].values.astype(np.float64)

                        # Use mid price for z-score calculation
                        mid = (bid_c + ask_c) / 2.0 if has_ba else bid_c.copy()

                        # Compute z-score
                        zscore = compute_rolling_zscore(mid, lb_bars)

                        # Generate trades
                        trades = generate_trades(
                            zscore, m5.index, bid_c, ask_c,
                            pname, pip_m, ez, exit_z, mh_bars)

                        if not trades:
                            continue

                        tdf = pd.DataFrame(trades)
                        key = f"{pname}|lb={lb_name}|ez={ez}|exit={exit_name}|mh={mh_name}"
                        label = f"{pname:<8} lb={lb_name:<4} z={ez:<3} exit={exit_name:<10} mh={mh_name}"

                        pnl = tdf['pnl_pips'].values
                        stats = compute_stats(pnl, label)
                        stats['pair'] = pname
                        stats['lookback'] = lb_name
                        stats['entry_z'] = ez
                        stats['exit_mode'] = exit_name
                        stats['max_hold'] = mh_name
                        stats['is_target'] = pname in TARGET_PAIRS
                        all_results.append(stats)
                        all_trades_by_key[key] = tdf

                        if combo_count % 20 == 0:
                            log.info(f"  {combo_count}/{total_combos} combos done...")

    log.info(f"  Grid search complete: {len(all_results)} combos with trades")

    if not all_results:
        log.error("No trades generated from any combo")
        sys.exit(1)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: RESULTS TABLE — ALL combos, sorted by avg pips/trade
    # ══════════════════════════════════════════════════════════════════

    # Sort by avg pips descending
    all_results.sort(key=lambda r: r['avg'], reverse=True)
    print_stats_table(all_results, "ALL PARAMETER COMBINATIONS — sorted by avg pips/trade")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: TARGET vs CONTROL comparison for best target config
    # ══════════════════════════════════════════════════════════════════

    # Find best config for target pairs
    target_results = [r for r in all_results if r['is_target'] and r['n'] > 20]
    control_results_list = [r for r in all_results if not r['is_target'] and r['n'] > 20]

    if target_results:
        best_target = max(target_results, key=lambda r: r['avg'])

        # Find same config on control pairs
        best_lb = best_target['lookback']
        best_ez = best_target['entry_z']
        best_exit = best_target['exit_mode']
        best_mh = best_target['max_hold']

        same_config_target = [r for r in all_results if r['is_target']
                              and r['lookback'] == best_lb
                              and r['entry_z'] == best_ez
                              and r['exit_mode'] == best_exit
                              and r['max_hold'] == best_mh]
        same_config_control = [r for r in all_results if not r['is_target']
                               and r['lookback'] == best_lb
                               and r['entry_z'] == best_ez
                               and r['exit_mode'] == best_exit
                               and r['max_hold'] == best_mh]

        run_legitimacy_tests(same_config_target, same_config_control)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: YEARLY BREAKDOWN for top configs
    # ══════════════════════════════════════════════════════════════════

    print_header("PHASE 4 — YEARLY BREAKDOWN (top 5 configs)")

    for r in all_results[:5]:
        lb = r['lookback']
        ez = r['entry_z']
        em = r['exit_mode']
        mh = r['max_hold']
        pname = r['pair']
        key = f"{pname}|lb={lb}|ez={ez}|exit={em}|mh={mh}"
        if key in all_trades_by_key:
            print_yearly(all_trades_by_key[key], r['label'])

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: LOSING MONTHS for best target config
    # ══════════════════════════════════════════════════════════════════

    if target_results:
        print_header("PHASE 5 — LOSING MONTHS (best target config)")
        best = max(target_results, key=lambda r: r['avg'])
        key = f"{best['pair']}|lb={best['lookback']}|ez={best['entry_z']}|exit={best['exit_mode']}|mh={best['max_hold']}"
        if key in all_trades_by_key:
            tdf = all_trades_by_key[key]
            tdf_copy = tdf.copy()
            tdf_copy['entry_dt'] = pd.to_datetime(tdf_copy['entry_ns'], unit='ns')
            tdf_copy['ym'] = tdf_copy['entry_dt'].dt.to_period('M')
            monthly = tdf_copy.groupby('ym')['pnl_pips'].sum()
            losing = (monthly < 0).sum()
            total_months = len(monthly)
            print(f"\n  {best['label']}")
            print(f"  Losing months: {losing} out of {total_months} "
                  f"({losing/total_months*100:.1f}%)")
            print(f"  Worst month: {monthly.min():+,.1f} pips")
            print(f"  Best month:  {monthly.max():+,.1f} pips")
            print(f"  Median month: {monthly.median():+,.1f} pips")

            # Show monthly P&L
            print(f"\n  {'Month':>10} {'PnL':>10} {'Trades':>7}")
            for ym in monthly.index:
                g = tdf_copy[tdf_copy['ym'] == ym]
                marker = "+" if monthly[ym] > 0 else "-"
                print(f"  {str(ym):>10} {monthly[ym]:>+10,.1f} {len(g):>7,}  {marker}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: SLIPPAGE + STREAK for best target config
    # ══════════════════════════════════════════════════════════════════

    if target_results:
        print_header("PHASE 6 — ROBUSTNESS TESTS")
        for r in target_results[:3]:  # top 3 target configs
            key = f"{r['pair']}|lb={r['lookback']}|ez={r['entry_z']}|exit={r['exit_mode']}|mh={r['max_hold']}"
            if key in all_trades_by_key:
                tdf = all_trades_by_key[key]
                run_slippage_test(tdf, r['label'])
                run_streak_analysis(tdf, r['label'])

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: EXIT REASON BREAKDOWN
    # ══════════════════════════════════════════════════════════════════

    if target_results:
        print_header("PHASE 7 — EXIT REASON BREAKDOWN")
        for r in target_results[:3]:
            key = f"{r['pair']}|lb={r['lookback']}|ez={r['entry_z']}|exit={r['exit_mode']}|mh={r['max_hold']}"
            if key in all_trades_by_key:
                tdf = all_trades_by_key[key]
                print(f"\n  {r['label']}")
                for reason, g in tdf.groupby('exit_reason'):
                    pnl = g['pnl_pips'].values
                    t = pnl.sum()
                    a = pnl.mean()
                    w = (pnl > 0).mean() * 100
                    ah = g['hold_hours'].mean()
                    print(f"    {reason:<12} {len(g):>6,} trades  "
                          f"{t:>+10,.1f} pips  avg {a:>+.3f}  "
                          f"WR {w:.1f}%  avg hold {ah:.1f}h")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: HOUR-OF-DAY ANALYSIS
    # ══════════════════════════════════════════════════════════════════

    if target_results:
        print_header("PHASE 8 — HOUR OF DAY (best target config)")
        best = max(target_results, key=lambda r: r['avg'])
        key = f"{best['pair']}|lb={best['lookback']}|ez={best['entry_z']}|exit={best['exit_mode']}|mh={best['max_hold']}"
        if key in all_trades_by_key:
            tdf = all_trades_by_key[key]
            tdf_copy = tdf.copy()
            tdf_copy['entry_dt'] = pd.to_datetime(tdf_copy['entry_ns'], unit='ns')
            tdf_copy['entry_hour'] = tdf_copy['entry_dt'].dt.hour

            print(f"\n  {best['label']}")
            print(f"  {'Hour':>6} {'Trades':>7} {'Total':>10} {'Avg':>8} "
                  f"{'WR':>6} {'Avg Spread':>10}")
            for h in range(24):
                g = tdf_copy[tdf_copy['entry_hour'] == h]
                if len(g) == 0:
                    continue
                pnl = g['pnl_pips'].values
                print(f"  {h:>6} {len(g):>7,} {pnl.sum():>+10,.1f} "
                      f"{pnl.mean():>+8.3f} {(pnl>0).mean()*100:>5.1f}% "
                      f"{g['entry_spread_pips'].mean():>10.2f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 9: Z-SCORE DISTRIBUTION AT ENTRY
    # ══════════════════════════════════════════════════════════════════

    if target_results:
        print_header("PHASE 9 — ENTRY Z-SCORE vs PnL (is deeper = better?)")
        best = max(target_results, key=lambda r: r['avg'])
        key = f"{best['pair']}|lb={best['lookback']}|ez={best['entry_z']}|exit={best['exit_mode']}|mh={best['max_hold']}"
        if key in all_trades_by_key:
            tdf = all_trades_by_key[key]
            abs_z = tdf['entry_z'].abs()
            print(f"\n  {best['label']}")
            print(f"  {'Z range':>15} {'Trades':>7} {'Avg PnL':>10} {'WR':>6}")
            for lo, hi in [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0),
                           (3.0, 4.0), (4.0, 99.0)]:
                mask = (abs_z >= lo) & (abs_z < hi)
                g = tdf[mask]
                if len(g) == 0:
                    continue
                pnl = g['pnl_pips'].values
                print(f"  {lo:.1f} - {hi:.1f}:  {len(g):>7,} "
                      f"{pnl.mean():>+10.3f} {(pnl>0).mean()*100:>5.1f}%")

    # ══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════

    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(output_dir, 'coint_backtest_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log.info(f"  Summary saved: {summary_path}")

    # Save trades for best target config
    if target_results:
        best = max(target_results, key=lambda r: r['avg'])
        key = f"{best['pair']}|lb={best['lookback']}|ez={best['entry_z']}|exit={best['exit_mode']}|mh={best['max_hold']}"
        if key in all_trades_by_key:
            trades_path = os.path.join(output_dir, 'coint_backtest_best_trades.csv')
            all_trades_by_key[key].to_csv(trades_path, index=False)
            log.info(f"  Best trades saved: {trades_path}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════

    elapsed = time_mod.time() - t_start

    print_header("FINAL SUMMARY")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Combos tested: {len(all_results)}")
    print(f"  Pairs tested: {len(pairs)}")

    if target_results:
        profitable_target = [r for r in target_results if r['avg'] > 0]
        profitable_control = [r for r in control_results_list if r['avg'] > 0]
        print(f"\n  Target combos profitable: {len(profitable_target)}/{len(target_results)}")
        print(f"  Control combos profitable: {len(profitable_control)}/{len(control_results_list)}")

        if profitable_target:
            best = max(target_results, key=lambda r: r['avg'])
            print(f"\n  BEST TARGET CONFIG:")
            print(f"    {best['label']}")
            print(f"    {best['n']:,} trades | {best['total']:+,.1f} pips | "
                  f"avg {best['avg']:+.3f} | WR {best['wr']:.1f}% | "
                  f"PF {best['pf']:.2f} | Sharpe {best['sharpe']:.2f}")
        else:
            print(f"\n  NO PROFITABLE TARGET CONFIGS FOUND")
            print(f"  Mean-reversion on these pairs at M5 resolution may not have edge")
    else:
        print(f"\n  No target pair results to report")

    print(f"\n{'=' * 120}")


if __name__ == '__main__':
    main()
