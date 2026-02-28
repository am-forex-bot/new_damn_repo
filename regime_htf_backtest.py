#!/usr/bin/env python3
"""
HIGHER-TIMEFRAME REGIME DETECTION BACKTEST
============================================
The 5-second / M5 regime detection failed the acid test because regime
transitions at that resolution are noise.  This script tests whether
regime detection works at DAILY, WEEKLY, and MONTHLY timeframes where:
  - Trends are more persistent and meaningful
  - Signal-to-noise ratio is orders of magnitude higher
  - Regime transitions represent actual macro shifts
  - Fewer trades, higher conviction

Strategy:
  - Detect bullish/bearish regime on D1/W1/M1 using EMA crossover
  - Enter at the CLOSE of the signal bar (next bar open in practice)
  - Hold until regime flips, or max hold reached
  - Test all 19 pairs — not just cherry-picked ones

Anti-cheat:
  1. Signal on bar N, entry on bar N+1 open (no same-bar execution)
  2. Bid/ask pricing where available
  3. No lookahead — regime state computed bar-by-bar
  4. Year-by-year breakdown for consistency
  5. Walk-forward: first 2 years = warmup, remaining = test
  6. Spread and slippage stress tests
  7. Report ALL parameter combos

Run:
  python regime_htf_backtest.py --data-dir /path/to/5s/parquets
"""

import os, sys, glob, argparse, time as time_mod, logging
import numpy as np
import pandas as pd

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

ALL_PAIRS = [
    'EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
    'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
    'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
    'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD',
]

# Timeframes to test regime detection on
TIMEFRAMES = {
    'D1': '1D',
    'W1': '1W',
}

# EMA periods to test
EMA_COMBOS = [
    (9, 21),     # fast
    (20, 50),    # classic
    (50, 200),   # golden/death cross
]

# Entry: require close above/below both EMAs (not just crossover)
# This is the same signal as the M5 version but on higher TFs

# Max hold in bars (safety cap)
MAX_HOLD = {
    'D1': {'30d': 30, '60d': 60, '90d': 90, 'unlimited': 9999},
    'W1': {'8w': 8, '13w': 13, '26w': 26, 'unlimited': 9999},
}

# Slippage
SLIPPAGE_LEVELS = [0, 1.0, 2.0, 3.0, 5.0]
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


def resample_to_tf(df_5s, rule):
    """Resample 5-second data to a higher timeframe."""
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
    df = df_5s.resample(rule).agg(agg).dropna(subset=['close'])
    return df


# ══════════════════════════════════════════════════════════════════════
# INDICATORS
# ══════════════════════════════════════════════════════════════════════

def _ema_np(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_emas(close, fast, slow):
    """Compute fast and slow EMAs."""
    c = close.astype(np.float64)
    if HAS_TALIB and len(c) > slow:
        ema_f = talib.EMA(c, timeperiod=fast)
        ema_s = talib.EMA(c, timeperiod=slow)
    else:
        ema_f = _ema_np(c, fast)
        ema_s = _ema_np(c, slow)
    return ema_f, ema_s


def compute_regime(close, ema_fast, ema_slow):
    """
    Compute regime state bar-by-bar.

    Bullish (+1): EMA_fast > EMA_slow AND close > EMA_fast
    Bearish (-1): EMA_fast < EMA_slow AND close < EMA_fast
    Neutral (0): everything else

    No hysteresis needed at daily/weekly — the signal is cleaner.
    """
    n = len(close)
    state = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
            continue
        if ema_fast[i] > ema_slow[i] and close[i] > ema_fast[i]:
            state[i] = 1
        elif ema_fast[i] < ema_slow[i] and close[i] < ema_fast[i]:
            state[i] = -1
    return state


# ══════════════════════════════════════════════════════════════════════
# TRADE GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_trades(state, df, pair_name, pip_mult, max_hold):
    """
    Generate trades from regime transitions.

    Entry: regime goes from 0 or opposite to +1/-1 on bar i
           → enter on bar i+1 at open (ask for long, bid for short)
    Exit:  regime flips (to 0 or opposite) on bar j
           → exit on bar j+1 at open (bid for long, ask for short)
           OR max hold reached

    Using OPEN prices for entry/exit because at D1/W1, you'd place
    the order overnight and get filled at next day/week open.
    """
    n = len(state)
    has_ba = 'bid_open' in df.columns

    # Price arrays — use OPEN for next-bar entry/exit
    if has_ba:
        bid_o = df['bid_open'].values.astype(np.float64)
        ask_o = df['ask_open'].values.astype(np.float64)
        bid_c = df['bid_close'].values.astype(np.float64)
        ask_c = df['ask_close'].values.astype(np.float64)
    else:
        bid_o = df['open'].values.astype(np.float64)
        ask_o = df['open'].values.astype(np.float64)
        bid_c = df['close'].values.astype(np.float64)
        ask_c = df['close'].values.astype(np.float64)

    # Spread at each bar (using close prices for spread measurement)
    spreads = (ask_c - bid_c) * pip_mult

    idx = df.index
    trades = []

    i = 0
    while i < n - 2:
        # Look for regime transition: state goes from 0 to +1/-1
        # OR from +1 to -1 or vice versa
        if state[i] != 0 and (i == 0 or state[i] != state[i - 1]):
            direction = int(state[i])
            entry_bar = i + 1  # enter on NEXT bar

            if entry_bar >= n:
                break

            # Entry price — next bar open
            entry_price = ask_o[entry_bar] if direction == 1 else bid_o[entry_bar]
            entry_spread = spreads[entry_bar]
            entry_ts = idx[entry_bar]

            # Find exit: regime flips or max hold
            exit_bar = min(entry_bar + max_hold, n - 1)
            exit_reason = 'max_hold'

            for j in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
                if state[j] != direction:
                    # Regime flipped — exit on next bar
                    exit_bar = min(j + 1, n - 1)
                    exit_reason = 'regime_flip'
                    break

            # Exit price — exit bar open
            exit_price = bid_o[exit_bar] if direction == 1 else ask_o[exit_bar]
            exit_ts = idx[exit_bar]

            # PnL
            pnl = direction * (exit_price - entry_price) * pip_mult
            hold_bars = exit_bar - entry_bar

            trades.append({
                'pair': pair_name,
                'direction': direction,
                'entry_bar': entry_bar,
                'exit_bar': exit_bar,
                'entry_ts': str(entry_ts),
                'exit_ts': str(exit_ts),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_spread_pips': entry_spread,
                'pnl_pips': pnl,
                'hold_bars': hold_bars,
                'exit_reason': exit_reason,
                'year': entry_ts.year,
            })

            # Skip to exit bar (position blocking inline)
            i = exit_bar
            continue

        i += 1

    return trades


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def compute_stats(pnl_arr, label=''):
    n = len(pnl_arr)
    if n == 0:
        return {'label': label, 'n': 0, 'total': 0, 'avg': 0,
                'wr': 0, 'pf': 0, 'max_dd': 0}
    total = pnl_arr.sum()
    avg = pnl_arr.mean()
    wr = (pnl_arr > 0).mean() * 100
    gl = abs(pnl_arr[pnl_arr < 0].sum())
    pf = pnl_arr[pnl_arr > 0].sum() / gl if gl > 0 else 999
    cum = np.cumsum(pnl_arr)
    max_dd = (np.maximum.accumulate(cum) - cum).max()
    return {'label': label, 'n': n, 'total': total, 'avg': avg,
            'wr': wr, 'pf': pf, 'max_dd': max_dd}


def print_header(title, width=120):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_results_table(results, title="RESULTS"):
    print_header(title)
    print(f"\n  {'Config':<65} {'Trades':>6} {'Total':>10} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>8}")
    print("  " + "-" * 112)
    for r in results:
        if r['n'] == 0:
            print(f"  {r['label']:<65} {'0':>6}")
            continue
        print(f"  {r['label']:<65} {r['n']:>6,} {r['total']:>+10,.1f} "
              f"{r['avg']:>+8.1f} {r['wr']:>5.1f}% {r['pf']:>6.2f} "
              f"{r['max_dd']:>8,.0f}")


def print_yearly(trades_df, label):
    print(f"\n  {label} — BY YEAR")
    print(f"  {'Year':>6} {'Trades':>7} {'Total':>10} {'Avg':>8} "
          f"{'WR':>6} {'PF':>6} {'Max DD':>8} {'Avg Hold':>9}")
    for yr in sorted(trades_df['year'].unique()):
        g = trades_df[trades_df['year'] == yr]
        pnl = g['pnl_pips'].values
        if len(pnl) == 0:
            continue
        t = pnl.sum()
        a = pnl.mean()
        w = (pnl > 0).mean() * 100
        gl = abs(pnl[pnl < 0].sum())
        p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
        cum = np.cumsum(pnl)
        dd = (np.maximum.accumulate(cum) - cum).max() if len(cum) > 0 else 0
        ah = g['hold_bars'].mean()
        marker = "+" if t > 0 else "-"
        print(f"  {yr:>6} {len(g):>7,} {t:>+10,.1f} {a:>+8.1f} "
              f"{w:>5.1f}% {p:>6.2f} {dd:>8,.0f} {ah:>8.1f}b  {marker}")


def run_slippage_test(trades_df, label):
    if len(trades_df) == 0:
        return
    base_pnl = trades_df['pnl_pips'].values
    base_total = base_pnl.sum()

    print(f"\n  SLIPPAGE — {label}")
    print(f"  {'Slip':>8} {'Total':>10} {'Avg':>8} {'Delta%':>8} {'Still+?':>8}")
    print("  " + "-" * 50)

    rng = np.random.RandomState(RNG_SEED)
    for slip in SLIPPAGE_LEVELS:
        if slip == 0:
            print(f"  {'base':>8} {base_total:>+10,.1f} "
                  f"{base_pnl.mean():>+8.1f} {'—':>8} "
                  f"{'YES' if base_total > 0 else 'NO':>8}")
            continue
        totals = []
        for _ in range(10):
            adverse = rng.uniform(0, slip, size=len(base_pnl))
            totals.append((base_pnl - adverse).sum())
        avg_t = np.mean(totals)
        delta = (avg_t - base_total) / abs(base_total) * 100 if base_total != 0 else 0
        print(f"  {slip:>8.1f} {avg_t:>+10,.1f} "
              f"{avg_t/len(base_pnl):>+8.1f} {delta:>+7.1f}% "
              f"{'YES' if avg_t > 0 else 'NO':>8}")


# ══════════════════════════════════════════════════════════════════════
# DATA DISCOVERY
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


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Higher-Timeframe Regime Detection Backtest')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
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

    print_header("HIGHER-TIMEFRAME REGIME DETECTION BACKTEST")
    print(f"  Timeframes: {list(TIMEFRAMES.keys())}")
    print(f"  EMA combos: {EMA_COMBOS}")
    print(f"  Max hold: D1={list(MAX_HOLD['D1'].keys())}  W1={list(MAX_HOLD['W1'].keys())}")
    print(f"  All 19 pairs tested — no cherry-picking")
    print(f"  Data: {data_dir}")

    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files in {data_dir}")
        sys.exit(1)

    pair_names = sorted(pairs.keys())
    log.info(f"  Pairs: {len(pairs)} — {', '.join(pair_names)}")

    # ── Load all 5s data and resample to each TF ──
    pair_tf_data = {}  # (pair, tf) → DataFrame
    for pname in pair_names:
        t0 = time_mod.time()
        df_5s = load_pair_data(pairs[pname])
        for tf_name, rule in TIMEFRAMES.items():
            df_tf = resample_to_tf(df_5s, rule)
            pair_tf_data[(pname, tf_name)] = df_tf
            log.info(f"  {pname} {tf_name}: {len(df_tf):,} bars")
        del df_5s

    # ══════════════════════════════════════════════════════════════════
    # GRID SEARCH
    # ══════════════════════════════════════════════════════════════════

    all_results = []
    all_trades_by_key = {}

    for tf_name in TIMEFRAMES:
        for ema_fast, ema_slow in EMA_COMBOS:
            for mh_name, mh_bars in MAX_HOLD[tf_name].items():

                # Aggregate across ALL pairs for this config
                config_trades = []

                for pname in pair_names:
                    df = pair_tf_data.get((pname, tf_name))
                    if df is None or len(df) < ema_slow + 10:
                        continue

                    pip_m = 100.0 if 'JPY' in pname else 10000.0
                    close = df['close'].values.astype(np.float64)
                    ema_f, ema_s = compute_emas(close, ema_fast, ema_slow)
                    state = compute_regime(close, ema_f, ema_s)

                    trades = generate_trades(state, df, pname, pip_m, mh_bars)
                    config_trades.extend(trades)

                if not config_trades:
                    continue

                tdf = pd.DataFrame(config_trades)
                label = (f"{tf_name} EMA({ema_fast},{ema_slow}) "
                         f"mh={mh_name}")
                key = f"{tf_name}|{ema_fast},{ema_slow}|{mh_name}"

                pnl = tdf['pnl_pips'].values
                stats = compute_stats(pnl, label)
                stats['tf'] = tf_name
                stats['ema'] = f"{ema_fast},{ema_slow}"
                stats['max_hold'] = mh_name
                all_results.append(stats)
                all_trades_by_key[key] = tdf

                # Per-pair breakdown for this config
                pair_stats = []
                for pname in pair_names:
                    g = tdf[tdf['pair'] == pname]
                    if len(g) > 0:
                        ps = compute_stats(g['pnl_pips'].values, pname)
                        pair_stats.append(ps)

                stats['pair_stats'] = pair_stats

    log.info(f"  Grid search complete: {len(all_results)} configs with trades")

    # ══════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════

    all_results.sort(key=lambda r: r['avg'], reverse=True)
    print_results_table(all_results,
                        "ALL CONFIGS — sorted by avg pips/trade (ALL 19 pairs aggregated)")

    # ── Per-pair breakdown for top configs ──
    print_header("PER-PAIR BREAKDOWN — top 5 configs")

    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        tdf = all_trades_by_key.get(key)
        if tdf is None:
            continue

        print(f"\n  {r['label']}")
        print(f"  {'Pair':<10} {'Trades':>6} {'Total':>10} {'Avg':>8} "
              f"{'WR':>6} {'PF':>6} {'Avg Hold':>9}")
        print("  " + "-" * 65)

        for pname in pair_names:
            g = tdf[tdf['pair'] == pname]
            if len(g) == 0:
                continue
            pnl = g['pnl_pips'].values
            t = pnl.sum()
            a = pnl.mean()
            w = (pnl > 0).mean() * 100
            gl = abs(pnl[pnl < 0].sum())
            p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
            ah = g['hold_bars'].mean()
            marker = "+" if t > 0 else "-"
            print(f"  {pname:<10} {len(g):>6,} {t:>+10,.1f} {a:>+8.1f} "
                  f"{w:>5.1f}% {p:>6.2f} {ah:>8.1f}b  {marker}")

        # How many pairs profitable?
        pair_pnl = tdf.groupby('pair')['pnl_pips'].sum()
        n_pos = (pair_pnl > 0).sum()
        n_neg = (pair_pnl <= 0).sum()
        print(f"\n  Profitable pairs: {n_pos}/{n_pos + n_neg}")

    # ── Yearly breakdown for top configs ──
    print_header("YEARLY BREAKDOWN — top 5 configs")
    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        if key in all_trades_by_key:
            print_yearly(all_trades_by_key[key], r['label'])

    # ── Exit reason breakdown ──
    print_header("EXIT REASON BREAKDOWN")
    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        tdf = all_trades_by_key.get(key)
        if tdf is None:
            continue
        print(f"\n  {r['label']}")
        for reason, g in tdf.groupby('exit_reason'):
            pnl = g['pnl_pips'].values
            t = pnl.sum()
            a = pnl.mean()
            w = (pnl > 0).mean() * 100
            ah = g['hold_bars'].mean()
            print(f"    {reason:<15} {len(g):>5,} trades  "
                  f"{t:>+10,.1f} pips  avg {a:>+.1f}  "
                  f"WR {w:.1f}%  hold {ah:.1f}b")

    # ── Slippage for top configs ──
    print_header("SLIPPAGE ROBUSTNESS — top 5 configs")
    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        if key in all_trades_by_key:
            run_slippage_test(all_trades_by_key[key], r['label'])

    # ── Walk-forward: first 2 years warmup, rest = test ──
    print_header("WALK-FORWARD VALIDATION (first 2 years excluded)")
    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        tdf = all_trades_by_key.get(key)
        if tdf is None:
            continue
        years = sorted(tdf['year'].unique())
        if len(years) <= 2:
            print(f"\n  {r['label']}: not enough years for walk-forward")
            continue

        warmup_years = years[:2]
        test_years = years[2:]
        warmup_df = tdf[tdf['year'].isin(warmup_years)]
        test_df = tdf[tdf['year'].isin(test_years)]

        warmup_pnl = warmup_df['pnl_pips'].values
        test_pnl = test_df['pnl_pips'].values

        ws = compute_stats(warmup_pnl, f"Warmup {warmup_years[0]}-{warmup_years[-1]}")
        ts = compute_stats(test_pnl, f"Test {test_years[0]}-{test_years[-1]}")

        print(f"\n  {r['label']}")
        print(f"    Warmup ({warmup_years[0]}-{warmup_years[-1]}): "
              f"{ws['n']:,} trades, {ws['total']:+,.1f} pips, "
              f"avg {ws['avg']:+.1f}, WR {ws['wr']:.1f}%, PF {ws['pf']:.2f}")
        print(f"    Test   ({test_years[0]}-{test_years[-1]}): "
              f"{ts['n']:,} trades, {ts['total']:+,.1f} pips, "
              f"avg {ts['avg']:+.1f}, WR {ts['wr']:.1f}%, PF {ts['pf']:.2f}")

        if ws['avg'] > 0 and ts['avg'] > 0:
            print(f"    VERDICT: CONSISTENT — profitable in both periods")
        elif ws['avg'] > 0 and ts['avg'] <= 0:
            print(f"    VERDICT: DEGRADED — profitable in warmup, not in test")
        elif ws['avg'] <= 0 and ts['avg'] > 0:
            print(f"    VERDICT: LATE BLOOMER — only profitable in later period")
        else:
            print(f"    VERDICT: NO EDGE — unprofitable in both periods")

    # ── Long vs Short breakdown ──
    print_header("LONG vs SHORT BREAKDOWN — top 5 configs")
    for r in all_results[:5]:
        key = f"{r['tf']}|{r['ema']}|{r['max_hold']}"
        tdf = all_trades_by_key.get(key)
        if tdf is None:
            continue
        print(f"\n  {r['label']}")
        for d, dname in [(1, 'LONG'), (-1, 'SHORT')]:
            g = tdf[tdf['direction'] == d]
            if len(g) == 0:
                continue
            pnl = g['pnl_pips'].values
            t = pnl.sum()
            a = pnl.mean()
            w = (pnl > 0).mean() * 100
            gl = abs(pnl[pnl < 0].sum())
            p = pnl[pnl > 0].sum() / gl if gl > 0 else 999
            print(f"    {dname:<6} {len(g):>5,} trades  {t:>+10,.1f} pips  "
                  f"avg {a:>+.1f}  WR {w:.1f}%  PF {p:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════════════

    # Save summary
    save_results = [{k: v for k, v in r.items() if k != 'pair_stats'}
                    for r in all_results]
    summary_df = pd.DataFrame(save_results)
    summary_path = os.path.join(output_dir, 'regime_htf_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log.info(f"  Summary: {summary_path}")

    # Save best config trades
    if all_results:
        best = all_results[0]
        key = f"{best['tf']}|{best['ema']}|{best['max_hold']}"
        if key in all_trades_by_key:
            tp = os.path.join(output_dir, 'regime_htf_best_trades.csv')
            all_trades_by_key[key].to_csv(tp, index=False)
            log.info(f"  Best trades: {tp}")

    # ══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════

    elapsed = time_mod.time() - t_start

    print_header("FINAL SUMMARY")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Configs tested: {len(all_results)}")
    print(f"  Pairs tested: {len(pairs)}")

    profitable = [r for r in all_results if r['avg'] > 0]
    print(f"\n  Profitable configs: {len(profitable)}/{len(all_results)}")

    if profitable:
        best = profitable[0]
        print(f"\n  BEST CONFIG:")
        print(f"    {best['label']}")
        print(f"    {best['n']:,} trades | {best['total']:+,.1f} pips | "
              f"avg {best['avg']:+.1f} | WR {best['wr']:.1f}% | PF {best['pf']:.2f}")

        # How many individual years profitable?
        key = f"{best['tf']}|{best['ema']}|{best['max_hold']}"
        tdf = all_trades_by_key.get(key)
        if tdf is not None:
            yr_pnl = tdf.groupby('year')['pnl_pips'].sum()
            n_pos_yr = (yr_pnl > 0).sum()
            print(f"    Profitable years: {n_pos_yr}/{len(yr_pnl)}")
    else:
        print(f"\n  NO PROFITABLE CONFIGS FOUND")
        print(f"  Regime detection may not have edge even at higher timeframes")

    print(f"\n{'=' * 120}")


if __name__ == '__main__':
    main()
