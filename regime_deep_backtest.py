#!/usr/bin/env python3
"""
REGIME DEEP BACKTEST — Multi-Layer Quantitative Regime Strategy
================================================================
Goes beyond simple EMA crossovers. Tests whether layering real quant
filters on top of regime detection produces a genuine, robust edge.

LAYERS TESTED (all gridded on/off):
  1. D1 trend alignment — only trade with the weekly/daily trend
  2. ADX trend strength — only enter when H1 ADX confirms a real trend
  3. Hurst exponent — only trade when market is actually persistent (H>0.5)
  4. Volatility-of-Volatility — skip chaotic VoV spikes
  5. RSI divergence block — don't enter when momentum diverges from price
  6. Pullback entry — wait for pullback to M5 EMA instead of chasing
  7. ATR trailing stop — adaptive exits instead of dumb regime flip
  8. Partial exit at 1R — lock in profits, let remainder run
  9. Regime-tighten exit — use regime flip to tighten stop, not instant close

Anti-cheat: same rigor as acid tests. Bid/ask, no lookahead, walk-forward,
slippage stress, year-by-year, all 19 pairs, component contribution analysis.

Run:
  python regime_deep_backtest.py --data-dir /path/to/5s/parquets
"""

import os, sys, glob, argparse, time as time_mod, logging, itertools
import numpy as np
import pandas as pd

try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Provide no-op decorator
    def njit(*args, **kwargs):
        def wrapper(f): return f
        if args and callable(args[0]): return args[0]
        return wrapper
    def prange(*a): return range(*a)

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

# Base regime detection (fixed — proven from acid tests)
MTF_WEIGHTS = {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}
ENTRY_THRESH = 0.4
EXIT_THRESH  = 0.2

# Fixed filters (always on — proven from acid tests)
MAX_SPREAD_PIPS = 5.0
TOXIC_HOURS_UTC = {20, 21, 22}
MAX_HOLD_BARS   = 576  # 48h in M5 bars

# Pullback max wait (in M5 bars) before giving up
PULLBACK_MAX_WAIT = 48  # 4 hours

# Grid dimensions
GRID = {
    'd1_filter':   ['off', 'ema_20_50', 'ema_50_200'],          # 3
    'adx_gate':    [0, 20, 25],                                  # 3
    'hurst_gate':  [0.0, 0.53],                                  # 2
    'vov_gate':    ['off', 'reject_p75'],                        # 2
    'rsi_div':     ['off', 'on'],                                # 2
    'entry':       ['immediate', 'pullback'],                    # 2
    'exit':        ['regime_flip', 'atr_2.5', 'atr_3.0',
                    'regime_tighten'],                            # 4
    'partial':     ['off', '50pct_1R'],                          # 2
}
# Total: 3×3×2×2×2×2×4×2 = 1,152 combos

SLIPPAGE_LEVELS = [0, 0.5, 1.0, 1.5, 2.0, 3.0]
RNG_SEED = 42
MIN_TRADES_TRAIN = 20

# Nanosecond bar periods for lookahead prevention
_TF_NS = {
    'M1': 60_000_000_000, 'M5': 300_000_000_000,
    'M15': 900_000_000_000, 'H1': 3_600_000_000_000,
    'H4': 14_400_000_000_000, 'D1': 86_400_000_000_000,
}


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


def _resample(df_5s, rule):
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'volume' in df_5s.columns:
        agg['volume'] = 'sum'
    for c, f in [('bid_open','first'),('bid_high','max'),('bid_low','min'),
                 ('bid_close','last'),('ask_open','first'),('ask_high','max'),
                 ('ask_low','min'),('ask_close','last')]:
        if c in df_5s.columns:
            agg[c] = f
    df = df_5s.resample(rule).agg(agg).dropna(subset=['close'])
    return df


def build_all_tfs(df_5s):
    tfs = {}
    for label, rule in [('M1','1min'),('M5','5min'),('M15','15min'),
                         ('H1','1h'),('H4','4h')]:
        tfs[label] = _resample(df_5s, rule)
    tfs['D1'] = _resample(df_5s, '1D')
    return tfs


def discover_pairs(data_dir):
    pairs = {}
    for ext in ['*.parquet', '*.pkl']:
        for f in glob.glob(os.path.join(data_dir, ext)):
            base = os.path.basename(f).upper()
            for p in ALL_PAIRS:
                if (p in base or p.replace('_','') in base) and p not in pairs:
                    pairs[p] = f
    return pairs


# ══════════════════════════════════════════════════════════════════════
# INDICATORS — all pure numpy, JIT where possible
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _ema(arr, span):
    alpha = 2.0 / (span + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def ema(arr, span):
    c = arr.astype(np.float64)
    if HAS_TALIB and len(c) > span:
        return talib.EMA(c, timeperiod=span)
    return _ema(c, span)


@njit(cache=True)
def _atr_wilder(high, low, close, period):
    """Wilder-smoothed ATR."""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    # First ATR
    s = 0.0
    for i in range(1, min(period + 1, n)):
        s += tr[i]
    if period < n:
        atr[period] = s / period
    for i in range(period + 1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


@njit(cache=True)
def _adx(high, low, close, period):
    """Full ADX computation: returns (atr, adx, +DI, -DI)."""
    n = len(close)
    tr = np.zeros(n); pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i]-low[i], max(abs(high[i]-close[i-1]), abs(low[i]-close[i-1])))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0: pdm[i] = up
        if down > up and down > 0: mdm[i] = down
    # Wilder smooth
    atr_s = 0.0; pdm_s = 0.0; mdm_s = 0.0
    atr = np.zeros(n); pdi = np.zeros(n); mdi = np.zeros(n)
    for i in range(1, min(period+1, n)):
        atr_s += tr[i]; pdm_s += pdm[i]; mdm_s += mdm[i]
    if period < n:
        atr[period] = atr_s / period
        if atr_s > 0:
            pdi[period] = 100.0 * pdm_s / atr_s
            mdi[period] = 100.0 * mdm_s / atr_s
    for i in range(period+1, n):
        atr_s = atr_s - atr_s/period + tr[i]
        pdm_s = pdm_s - pdm_s/period + pdm[i]
        mdm_s = mdm_s - mdm_s/period + mdm[i]
        atr[i] = atr_s / period
        if atr_s > 0:
            pdi[i] = 100.0 * pdm_s / atr_s
            mdi[i] = 100.0 * mdm_s / atr_s
    # DX → ADX
    dx = np.zeros(n); adx = np.zeros(n)
    for i in range(period, n):
        s = pdi[i] + mdi[i]
        if s > 0: dx[i] = 100.0 * abs(pdi[i] - mdi[i]) / s
    if 2*period <= n:
        s = 0.0
        for i in range(period, 2*period):
            s += dx[i]
        adx[2*period-1] = s / period
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return atr, adx, pdi, mdi


@njit(cache=True)
def _rsi(close, period):
    """Wilder-smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    ag = 0.0; al = 0.0
    for i in range(1, period + 1):
        d = close[i] - close[i-1]
        if d > 0: ag += d
        else: al -= d
    ag /= period; al /= period
    if al > 0: rsi[period] = 100.0 - 100.0 / (1.0 + ag / al)
    elif ag > 0: rsi[period] = 100.0
    for i in range(period + 1, n):
        d = close[i] - close[i-1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        ag = (ag * (period - 1) + g) / period
        al = (al * (period - 1) + l) / period
        if al > 0: rsi[i] = 100.0 - 100.0 / (1.0 + ag / al)
        elif ag > 0: rsi[i] = 100.0
        else: rsi[i] = 50.0
    return rsi


@njit(cache=True)
def _hurst_rs(close, window):
    """Rolling Hurst exponent via rescaled range (R/S) analysis."""
    n = len(close)
    hurst = np.full(n, 0.5)  # default = random walk
    log_w = np.log(float(window))
    if log_w < 1e-10:
        return hurst
    for i in range(window, n):
        # Log returns over window
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += close[j] - close[j-1]
        mean /= window
        # Cumulative deviation
        cum = 0.0; mn = 0.0; mx = 0.0; ss = 0.0
        for j in range(i - window + 1, i + 1):
            ret = (close[j] - close[j-1]) - mean
            cum += ret
            if cum < mn: mn = cum
            if cum > mx: mx = cum
            ss += ret * ret
        R = mx - mn
        S = np.sqrt(ss / window)
        if S > 1e-15 and R > 1e-15:
            hurst[i] = np.log(R / S) / log_w
    return hurst


@njit(cache=True)
def _rolling_std(arr, window):
    """Rolling standard deviation."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window - 1, n):
        s = 0.0; s2 = 0.0
        for j in range(i - window + 1, i + 1):
            s += arr[j]; s2 += arr[j] * arr[j]
        mean = s / window
        var = s2 / window - mean * mean
        out[i] = np.sqrt(max(var, 0.0))
    return out


@njit(cache=True)
def _rolling_pctile_rank(arr, window):
    """Rolling percentile rank of current value within its own history."""
    n = len(arr)
    rank = np.full(n, 50.0)
    for i in range(window, n):
        val = arr[i]; below = 0; valid = 0
        for j in range(i - window, i):
            if arr[j] == arr[j]:  # not NaN
                valid += 1
                if arr[j] < val: below += 1
        if valid > 0:
            rank[i] = 100.0 * below / valid
    return rank


@njit(cache=True)
def _rsi_divergence(close, rsi_arr, lookback):
    """
    Detect RSI divergence via linear regression slope comparison.
    Returns: +1 bullish div, -1 bearish div, 0 none.
    """
    n = len(close)
    div = np.zeros(n, dtype=np.int32)
    x_sum = 0.0; x2_sum = 0.0
    for j in range(lookback):
        x_sum += j; x2_sum += j * j
    denom = lookback * x2_sum - x_sum * x_sum
    if abs(denom) < 1e-15:
        return div
    for i in range(lookback, n):
        xy_p = 0.0; y_p = 0.0; xy_r = 0.0; y_r = 0.0
        for j in range(lookback):
            idx = i - lookback + 1 + j
            xy_p += j * close[idx]; y_p += close[idx]
            xy_r += j * rsi_arr[idx]; y_r += rsi_arr[idx]
        slope_p = (lookback * xy_p - x_sum * y_p) / denom
        slope_r = (lookback * xy_r - x_sum * y_r) / denom
        # Meaningful divergence: slopes opposite with some magnitude
        price_range = 0.0
        for j in range(lookback):
            idx = i - lookback + 1 + j
            price_range = max(price_range, abs(close[idx] - close[i]))
        if price_range < 1e-10:
            continue
        norm_slope = slope_p / (price_range / lookback + 1e-15)
        if norm_slope > 0.1 and slope_r < -0.3:
            div[i] = -1  # bearish: price up, RSI down
        elif norm_slope < -0.1 and slope_r > 0.3:
            div[i] = 1   # bullish: price down, RSI up
    return div


# ══════════════════════════════════════════════════════════════════════
# MTF BIAS + REGIME STATE MACHINE (from existing system)
# ══════════════════════════════════════════════════════════════════════

def add_emas(df):
    if len(df) < 22: return
    c = df['close'].values.astype(np.float64)
    df['ema_9'] = ema(c, 9)
    df['ema_21'] = ema(c, 21)


def compute_mtf_bias(tfs, m5_index):
    total_w = sum(MTF_WEIGHTS.values())
    n = len(m5_index)
    bias = np.zeros(n, dtype=np.float64)
    m5_ns = m5_index.asi8
    for tf_name, w in MTF_WEIGHTS.items():
        df = tfs.get(tf_name)
        if df is None or len(df) < 22 or 'ema_9' not in df.columns:
            continue
        e9 = df['ema_9'].values; e21 = df['ema_21'].values; cl = df['close'].values
        sig = np.where((e9>e21)&(cl>e9), 1.0, np.where((e9<e21)&(cl<e9), -1.0, 0.0))
        tt = df.index.asi8
        if tf_name in ('M15', 'H1', 'H4'):
            tt = tt + _TF_NS[tf_name]
        idx = np.clip(np.searchsorted(tt, m5_ns, side='right') - 1, 0, len(sig) - 1)
        bias += sig[idx] * w
    if total_w > 0 and total_w != 1.0:
        bias /= total_w
    return np.clip(bias, -1.0, 1.0)


@njit(cache=True)
def _regime_hysteresis(bias, et, xt):
    n = len(bias)
    state = np.zeros(n, dtype=np.int32)
    cur = np.int32(0)
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


# ══════════════════════════════════════════════════════════════════════
# EXIT PRECOMPUTATION — Numba kernels for path-dependent exits
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _find_regime_exit(state, entry_bar, direction, max_hold, n_bars):
    """Find bar where regime flips off. Returns (exit_bar, exit_offset)."""
    for off in range(1, min(max_hold + 1, n_bars - entry_bar)):
        if state[entry_bar + off] != direction:
            return entry_bar + off
    return min(entry_bar + max_hold, n_bars - 1)


@njit(cache=True)
def _atr_trail_exit(direction, entry_bar, entry_price, high, low,
                    atr_mapped, mult, max_hold, n_bars):
    """ATR trailing stop. Returns (exit_bar, exit_price)."""
    if direction == 1:
        stop = entry_price - mult * atr_mapped[entry_bar]
        best = high[entry_bar]
        for off in range(1, min(max_hold + 1, n_bars - entry_bar)):
            b = entry_bar + off
            if high[b] > best: best = high[b]
            ns = best - mult * atr_mapped[b]
            if ns > stop: stop = ns
            if low[b] <= stop:
                return b, stop
    else:
        stop = entry_price + mult * atr_mapped[entry_bar]
        best = low[entry_bar]
        for off in range(1, min(max_hold + 1, n_bars - entry_bar)):
            b = entry_bar + off
            if low[b] < best: best = low[b]
            ns = best + mult * atr_mapped[b]
            if ns < stop: stop = ns
            if high[b] >= stop:
                return b, stop
    end = min(entry_bar + max_hold, n_bars - 1)
    return end, (low[end] if direction == 1 else high[end])


@njit(cache=True)
def _regime_tighten_exit(direction, entry_bar, entry_price, state,
                         high, low, atr_mapped, wide_mult, tight_mult,
                         max_hold, n_bars):
    """
    Start with wide ATR trail while regime ON.
    When regime flips OFF, tighten to narrow ATR trail.
    This gives the trade room to breathe but protects on regime weakness.
    """
    tightened = False
    if direction == 1:
        stop = entry_price - wide_mult * atr_mapped[entry_bar]
        best = high[entry_bar]
        for off in range(1, min(max_hold + 1, n_bars - entry_bar)):
            b = entry_bar + off
            if high[b] > best: best = high[b]
            # Check regime
            if not tightened and state[b] != direction:
                tightened = True
                # Tighten: recalculate from current best with tight multiplier
                tight_stop = best - tight_mult * atr_mapped[b]
                if tight_stop > stop: stop = tight_stop
            if tightened:
                ns = best - tight_mult * atr_mapped[b]
            else:
                ns = best - wide_mult * atr_mapped[b]
            if ns > stop: stop = ns
            if low[b] <= stop:
                return b, stop
    else:
        stop = entry_price + wide_mult * atr_mapped[entry_bar]
        best = low[entry_bar]
        for off in range(1, min(max_hold + 1, n_bars - entry_bar)):
            b = entry_bar + off
            if low[b] < best: best = low[b]
            if not tightened and state[b] != direction:
                tightened = True
                tight_stop = best + tight_mult * atr_mapped[b]
                if tight_stop < stop: stop = tight_stop
            if tightened:
                ns = best + tight_mult * atr_mapped[b]
            else:
                ns = best + wide_mult * atr_mapped[b]
            if ns < stop: stop = ns
            if high[b] >= stop:
                return b, stop
    end = min(entry_bar + max_hold, n_bars - 1)
    return end, (low[end] if direction == 1 else high[end])


@njit(cache=True)
def _find_pullback(direction, signal_bar, close, ema21, max_wait, n_bars):
    """Find pullback entry: price touches EMA then resumes. Returns bar or -1."""
    touched = False
    for off in range(1, min(max_wait + 1, n_bars - signal_bar)):
        b = signal_bar + off
        if direction == 1:
            if close[b] <= ema21[b]:
                touched = True
            elif touched and close[b] > ema21[b]:
                return b
        else:
            if close[b] >= ema21[b]:
                touched = True
            elif touched and close[b] < ema21[b]:
                return b
    return -1


@njit(cache=True)
def _find_1r_bar(direction, entry_bar, entry_price, risk_dist,
                 high, low, max_bar, n_bars):
    """Find first bar where 1R profit is reached. Returns bar or -1."""
    if risk_dist <= 0:
        return -1
    target = entry_price + direction * risk_dist
    for off in range(1, min(max_bar - entry_bar + 1, n_bars - entry_bar)):
        b = entry_bar + off
        if direction == 1 and high[b] >= target:
            return b
        elif direction == -1 and low[b] <= target:
            return b
    return -1


# ══════════════════════════════════════════════════════════════════════
# PER-PAIR PROCESSING — compute all indicators, extract events
# ══════════════════════════════════════════════════════════════════════

def _map_htf_to_m5(htf_arr, htf_index, m5_ns, tf_name):
    """Map higher-TF indicator values to M5 bars with lookahead prevention."""
    tt = htf_index.asi8
    if tf_name in _TF_NS:
        tt = tt + _TF_NS[tf_name]
    idx = np.clip(np.searchsorted(tt, m5_ns, side='right') - 1, 0, len(htf_arr) - 1)
    return htf_arr[idx]


def process_pair(df_5s, pair_name, pair_idx, pip_mult):
    """
    Full processing for one pair:
    1. Resample to all TFs
    2. Compute all indicators
    3. Detect regime transitions
    4. For each event, evaluate filters and precompute all exit PnLs
    Returns list of event dicts with all precomputed values.
    """
    tfs = build_all_tfs(df_5s)
    m5 = tfs['M5']
    d1 = tfs['D1']
    h1 = tfs['H1']
    n_m5 = len(m5)

    if n_m5 < 500 or len(d1) < 200 or len(h1) < 50:
        log.warning(f"  {pair_name}: insufficient data (M5={n_m5}, D1={len(d1)}, H1={len(h1)})")
        return []

    # ── Add EMAs to all TFs for regime detection ──
    for tf in tfs.values():
        add_emas(tf)

    # ── M5 arrays ──
    m5_close = m5['close'].values.astype(np.float64)
    m5_high  = m5['high'].values.astype(np.float64)
    m5_low   = m5['low'].values.astype(np.float64)
    m5_ns    = m5.index.asi8
    m5_ema21 = ema(m5_close, 21)

    has_ba = 'bid_close' in m5.columns and 'ask_close' in m5.columns
    if has_ba:
        bid_c = m5['bid_close'].values.astype(np.float64)
        ask_c = m5['ask_close'].values.astype(np.float64)
    else:
        bid_c = m5_close.copy()
        ask_c = m5_close.copy()

    # ── D1 indicators ──
    d1_close = d1['close'].values.astype(np.float64)
    d1_high  = d1['high'].values.astype(np.float64)
    d1_low   = d1['low'].values.astype(np.float64)

    # D1 EMAs for trend filter
    d1_ema20 = ema(d1_close, 20)
    d1_ema50 = ema(d1_close, 50)
    d1_ema200 = ema(d1_close, 200)
    # D1 trend states: +1 bullish, -1 bearish, 0 neutral
    d1_trend_20_50 = np.where(d1_ema20 > d1_ema50, 1, np.where(d1_ema20 < d1_ema50, -1, 0)).astype(np.int32)
    d1_trend_50_200 = np.where(d1_ema50 > d1_ema200, 1, np.where(d1_ema50 < d1_ema200, -1, 0)).astype(np.int32)

    # D1 Hurst exponent (100-bar rolling R/S)
    d1_hurst = _hurst_rs(d1_close, 100)

    # D1 VoV: std of ATR, percentile ranked
    d1_atr = _atr_wilder(d1_high, d1_low, d1_close, 14)
    d1_vov = _rolling_std(d1_atr, 20)
    d1_vov_pctile = _rolling_pctile_rank(d1_vov, 100)

    # Map D1 indicators → M5
    m5_d1_trend_20_50  = _map_htf_to_m5(d1_trend_20_50, d1.index, m5_ns, 'D1')
    m5_d1_trend_50_200 = _map_htf_to_m5(d1_trend_50_200, d1.index, m5_ns, 'D1')
    m5_d1_hurst        = _map_htf_to_m5(d1_hurst, d1.index, m5_ns, 'D1')
    m5_d1_vov_pctile   = _map_htf_to_m5(d1_vov_pctile, d1.index, m5_ns, 'D1')

    # ── H1 indicators ──
    h1_close = h1['close'].values.astype(np.float64)
    h1_high  = h1['high'].values.astype(np.float64)
    h1_low   = h1['low'].values.astype(np.float64)

    h1_atr_arr, h1_adx_arr, _, _ = _adx(h1_high, h1_low, h1_close, 14)
    h1_rsi_arr = _rsi(h1_close, 14)
    h1_rsi_div = _rsi_divergence(h1_close, h1_rsi_arr, 14)

    # Map H1 indicators → M5
    m5_h1_adx     = _map_htf_to_m5(h1_adx_arr, h1.index, m5_ns, 'H1')
    m5_h1_atr     = _map_htf_to_m5(h1_atr_arr, h1.index, m5_ns, 'H1')
    m5_h1_rsi_div = _map_htf_to_m5(h1_rsi_div, h1.index, m5_ns, 'H1')

    # ── Regime detection ──
    bias = compute_mtf_bias(tfs, m5.index)
    state = _regime_hysteresis(bias, ENTRY_THRESH, EXIT_THRESH)

    # ── Extract regime transitions and build events ──
    events = []
    spread_arr = (ask_c - bid_c) * pip_mult

    for i in range(1, n_m5):
        if state[i-1] == 0 and state[i] != 0:
            ts = m5.index[i]
            # Weekend filter
            if ts.dayofweek >= 5:
                continue
            # Toxic hour filter
            if ts.hour in TOXIC_HOURS_UTC:
                continue
            # Spread filter (check at signal bar)
            if spread_arr[i] > MAX_SPREAD_PIPS:
                continue

            d = int(state[i])
            sig_bar = i

            # ── Filter values at this event ──
            ev = {
                'pair_idx': pair_idx,
                'pair': pair_name,
                'direction': d,
                'signal_bar': sig_bar,
                'year': ts.year,
                'dow': ts.dayofweek,
                'window': ts.hour * 2 + (1 if ts.minute >= 30 else 0),
                'entry_ts': str(ts),
                # Filter values
                'd1_trend_20_50_match':  (m5_d1_trend_20_50[sig_bar] == d),
                'd1_trend_50_200_match': (m5_d1_trend_50_200[sig_bar] == d),
                'adx_val':  float(m5_h1_adx[sig_bar]),
                'hurst_val': float(m5_d1_hurst[sig_bar]),
                'vov_pctile': float(m5_d1_vov_pctile[sig_bar]),
                'rsi_div_against': bool(m5_h1_rsi_div[sig_bar] == -d),
            }

            # ── Entry methods ──
            # 0: Immediate (signal bar)
            imm_bar = sig_bar
            imm_price = ask_c[imm_bar] if d == 1 else bid_c[imm_bar]

            # 1: Pullback to M5 EMA(21)
            pb_bar_raw = _find_pullback(d, sig_bar, m5_close, m5_ema21,
                                        PULLBACK_MAX_WAIT, n_m5)
            pb_bar = pb_bar_raw if pb_bar_raw > 0 else -1
            pb_price = (ask_c[pb_bar] if d == 1 else bid_c[pb_bar]) if pb_bar > 0 else 0.0

            ev['entry_bars'] = np.array([imm_bar, pb_bar], dtype=np.int64)
            ev['entry_prices'] = np.array([imm_price, pb_price], dtype=np.float64)
            ev['entry_valid'] = np.array([True, pb_bar > 0], dtype=np.bool_)

            # ── Exit precomputation for each entry method ──
            # PnL arrays: shape (2 entry, 4 exit)
            pnl_full = np.full((2, 4), np.nan, dtype=np.float64)
            pnl_partial = np.full((2, 4), np.nan, dtype=np.float64)
            exit_bars_out = np.full((2, 4), -1, dtype=np.int64)

            for ent_idx, (e_bar, e_price, e_valid) in enumerate(
                    zip([imm_bar, pb_bar], [imm_price, pb_price], [True, pb_bar > 0])):
                if not e_valid or e_bar >= n_m5:
                    continue

                atr_at_entry = m5_h1_atr[e_bar]
                if atr_at_entry <= 0:
                    atr_at_entry = abs(m5_close[e_bar]) * 0.001

                # EXIT 0: Regime flip
                ex0_bar = _find_regime_exit(state, e_bar, d, MAX_HOLD_BARS, n_m5)
                ex0_price = bid_c[ex0_bar] if d == 1 else ask_c[ex0_bar]
                pnl_full[ent_idx, 0] = d * (ex0_price - e_price) * pip_mult
                pnl_partial[ent_idx, 0] = pnl_full[ent_idx, 0]
                exit_bars_out[ent_idx, 0] = ex0_bar

                # EXIT 1: ATR trail 2.5×
                ex1_bar, ex1_price = _atr_trail_exit(
                    d, e_bar, e_price, m5_high, m5_low, m5_h1_atr,
                    2.5, MAX_HOLD_BARS, n_m5)
                pnl_full[ent_idx, 1] = d * (ex1_price - e_price) * pip_mult
                exit_bars_out[ent_idx, 1] = ex1_bar
                # Partial: 50% at 1R (risk = 2.5 × ATR)
                risk1 = 2.5 * atr_at_entry
                r1_bar = _find_1r_bar(d, e_bar, e_price, risk1,
                                       m5_high, m5_low, ex1_bar, n_m5)
                if r1_bar > 0 and r1_bar < ex1_bar:
                    pnl_partial[ent_idx, 1] = 0.5 * risk1 * pip_mult + \
                        0.5 * d * (ex1_price - e_price) * pip_mult
                else:
                    pnl_partial[ent_idx, 1] = pnl_full[ent_idx, 1]

                # EXIT 2: ATR trail 3.0×
                ex2_bar, ex2_price = _atr_trail_exit(
                    d, e_bar, e_price, m5_high, m5_low, m5_h1_atr,
                    3.0, MAX_HOLD_BARS, n_m5)
                pnl_full[ent_idx, 2] = d * (ex2_price - e_price) * pip_mult
                exit_bars_out[ent_idx, 2] = ex2_bar
                risk2 = 3.0 * atr_at_entry
                r2_bar = _find_1r_bar(d, e_bar, e_price, risk2,
                                       m5_high, m5_low, ex2_bar, n_m5)
                if r2_bar > 0 and r2_bar < ex2_bar:
                    pnl_partial[ent_idx, 2] = 0.5 * risk2 * pip_mult + \
                        0.5 * d * (ex2_price - e_price) * pip_mult
                else:
                    pnl_partial[ent_idx, 2] = pnl_full[ent_idx, 2]

                # EXIT 3: Regime tighten 3→1.5
                ex3_bar, ex3_price = _regime_tighten_exit(
                    d, e_bar, e_price, state, m5_high, m5_low, m5_h1_atr,
                    3.0, 1.5, MAX_HOLD_BARS, n_m5)
                pnl_full[ent_idx, 3] = d * (ex3_price - e_price) * pip_mult
                exit_bars_out[ent_idx, 3] = ex3_bar
                risk3 = 3.0 * atr_at_entry
                r3_bar = _find_1r_bar(d, e_bar, e_price, risk3,
                                       m5_high, m5_low, ex3_bar, n_m5)
                if r3_bar > 0 and r3_bar < ex3_bar:
                    pnl_partial[ent_idx, 3] = 0.5 * risk3 * pip_mult + \
                        0.5 * d * (ex3_price - e_price) * pip_mult
                else:
                    pnl_partial[ent_idx, 3] = pnl_full[ent_idx, 3]

            ev['pnl_full'] = pnl_full
            ev['pnl_partial'] = pnl_partial
            ev['exit_bars'] = exit_bars_out
            events.append(ev)

    log.info(f"  {pair_name}: {len(events):,} events from {n_m5:,} M5 bars")
    return events


# ══════════════════════════════════════════════════════════════════════
# GLOBAL EVENT ARRAYS — merge all pairs into vectorizable arrays
# ══════════════════════════════════════════════════════════════════════

def build_global_arrays(all_events):
    """
    Merge per-pair events into global numpy arrays for fast grid sweep.
    Returns dict of arrays, all shape (n_events, ...).
    """
    n = len(all_events)
    if n == 0:
        return None

    g = {
        'pair_idx':    np.array([e['pair_idx'] for e in all_events], dtype=np.int32),
        'pair':        [e['pair'] for e in all_events],
        'direction':   np.array([e['direction'] for e in all_events], dtype=np.int32),
        'year':        np.array([e['year'] for e in all_events], dtype=np.int32),
        'dow':         np.array([e['dow'] for e in all_events], dtype=np.int32),
        'window':      np.array([e['window'] for e in all_events], dtype=np.int32),
        # Filter booleans
        'd1_20_50':    np.array([e['d1_trend_20_50_match'] for e in all_events], dtype=np.bool_),
        'd1_50_200':   np.array([e['d1_trend_50_200_match'] for e in all_events], dtype=np.bool_),
        'adx':         np.array([e['adx_val'] for e in all_events], dtype=np.float32),
        'hurst':       np.array([e['hurst_val'] for e in all_events], dtype=np.float32),
        'vov_pctile':  np.array([e['vov_pctile'] for e in all_events], dtype=np.float32),
        'rsi_div_ag':  np.array([e['rsi_div_against'] for e in all_events], dtype=np.bool_),
        # Entry validity
        'entry_valid': np.stack([e['entry_valid'] for e in all_events]),   # (n, 2)
        # PnL arrays
        'pnl_full':    np.stack([e['pnl_full'] for e in all_events]),      # (n, 2, 4)
        'pnl_partial': np.stack([e['pnl_partial'] for e in all_events]),   # (n, 2, 4)
        'n_events':    n,
    }
    return g


# ══════════════════════════════════════════════════════════════════════
# GRID SWEEP — vectorized numpy
# ══════════════════════════════════════════════════════════════════════

# Grid indices
D1_OPTS   = [('off', None), ('ema_20_50', 'd1_20_50'), ('ema_50_200', 'd1_50_200')]
ADX_OPTS  = [0, 20, 25]
HURST_OPTS = [0.0, 0.53]
VOV_OPTS  = [('off', 999), ('reject_p75', 75)]
DIV_OPTS  = ['off', 'on']
ENTRY_OPTS = [0, 1]   # 0=immediate, 1=pullback
EXIT_OPTS  = ['regime_flip', 'atr_2.5', 'atr_3.0', 'regime_tighten']
PARTIAL_OPTS = ['off', '50pct_1R']

N_FILTER_COMBOS = len(D1_OPTS) * len(ADX_OPTS) * len(HURST_OPTS) * len(VOV_OPTS) * len(DIV_OPTS)
N_METHOD_COMBOS = len(ENTRY_OPTS) * len(EXIT_OPTS) * len(PARTIAL_OPTS)
N_TOTAL_COMBOS  = N_FILTER_COMBOS * N_METHOD_COMBOS


def _decode_combo(combo_idx):
    """Decode flat combo index into grid coordinates."""
    mc = combo_idx % N_METHOD_COMBOS
    fc = combo_idx // N_METHOD_COMBOS
    # Filter coords
    div_i = fc % len(DIV_OPTS); fc //= len(DIV_OPTS)
    vov_i = fc % len(VOV_OPTS); fc //= len(VOV_OPTS)
    hur_i = fc % len(HURST_OPTS); fc //= len(HURST_OPTS)
    adx_i = fc % len(ADX_OPTS); fc //= len(ADX_OPTS)
    d1_i  = fc % len(D1_OPTS)
    # Method coords
    par_i = mc % len(PARTIAL_OPTS); mc //= len(PARTIAL_OPTS)
    ext_i = mc % len(EXIT_OPTS); mc //= len(EXIT_OPTS)
    ent_i = mc % len(ENTRY_OPTS)
    return d1_i, adx_i, hur_i, vov_i, div_i, ent_i, ext_i, par_i


def _combo_label(d1_i, adx_i, hur_i, vov_i, div_i, ent_i, ext_i, par_i):
    parts = []
    parts.append(f"D1={D1_OPTS[d1_i][0]}")
    parts.append(f"ADX≥{ADX_OPTS[adx_i]}" if ADX_OPTS[adx_i] > 0 else "ADX=off")
    parts.append(f"H≥{HURST_OPTS[hur_i]:.2f}" if HURST_OPTS[hur_i] > 0 else "H=off")
    parts.append(f"VoV={VOV_OPTS[vov_i][0]}")
    parts.append(f"Div={DIV_OPTS[div_i]}")
    parts.append(f"Ent={'pullback' if ent_i else 'immed'}")
    parts.append(f"Ex={EXIT_OPTS[ext_i]}")
    parts.append(f"Part={PARTIAL_OPTS[par_i]}")
    return ' | '.join(parts)


def run_grid_sweep(g, years_list):
    """
    Sweep all 1,152 combos. For each combo, compute per-year PnL sums and trade counts.
    Returns (sums, counts) of shape (N_TOTAL_COMBOS, n_years).
    """
    n_events = g['n_events']
    n_years = len(years_list)
    year_map = {y: i for i, y in enumerate(years_list)}
    yr_idx = np.array([year_map.get(y, -1) for y in g['year']], dtype=np.int32)

    # Precompute filter masks: (n_filter_combos, n_events)
    log.info(f"  Building {N_FILTER_COMBOS} filter masks × {n_events:,} events ...")
    filter_masks = np.zeros((N_FILTER_COMBOS, n_events), dtype=np.bool_)

    fi = 0
    for d1_i in range(len(D1_OPTS)):
        if d1_i == 0:
            d1_mask = np.ones(n_events, dtype=np.bool_)
        elif d1_i == 1:
            d1_mask = g['d1_20_50']
        else:
            d1_mask = g['d1_50_200']
        for adx_i in range(len(ADX_OPTS)):
            adx_mask = g['adx'] >= ADX_OPTS[adx_i] if ADX_OPTS[adx_i] > 0 else \
                       np.ones(n_events, dtype=np.bool_)
            for hur_i in range(len(HURST_OPTS)):
                hur_mask = g['hurst'] >= HURST_OPTS[hur_i] if HURST_OPTS[hur_i] > 0 else \
                           np.ones(n_events, dtype=np.bool_)
                for vov_i in range(len(VOV_OPTS)):
                    vov_mask = g['vov_pctile'] < VOV_OPTS[vov_i][1]
                    for div_i in range(len(DIV_OPTS)):
                        div_mask = ~g['rsi_div_ag'] if div_i == 1 else \
                                   np.ones(n_events, dtype=np.bool_)
                        filter_masks[fi] = d1_mask & adx_mask & hur_mask & vov_mask & div_mask
                        fi += 1

    # Sweep: for each combo, sum PnLs per year
    sums = np.zeros((N_TOTAL_COMBOS, n_years), dtype=np.float64)
    counts = np.zeros((N_TOTAL_COMBOS, n_years), dtype=np.int32)

    log.info(f"  Sweeping {N_TOTAL_COMBOS:,} combos ...")
    for fi in range(N_FILTER_COMBOS):
        fmask = filter_masks[fi]
        for ent_i in range(len(ENTRY_OPTS)):
            # Combine filter mask with entry validity
            emask = fmask & g['entry_valid'][:, ent_i]
            valid_idx = np.where(emask)[0]
            if len(valid_idx) == 0:
                continue
            v_yr = yr_idx[valid_idx]
            for ext_i in range(len(EXIT_OPTS)):
                for par_i in range(len(PARTIAL_OPTS)):
                    combo_flat = (fi * N_METHOD_COMBOS +
                                  ent_i * len(EXIT_OPTS) * len(PARTIAL_OPTS) +
                                  ext_i * len(PARTIAL_OPTS) +
                                  par_i)
                    if par_i == 0:
                        pnl_arr = g['pnl_full'][valid_idx, ent_i, ext_i]
                    else:
                        pnl_arr = g['pnl_partial'][valid_idx, ent_i, ext_i]
                    # Remove NaN (invalid exits)
                    valid_pnl = ~np.isnan(pnl_arr)
                    for yi in range(n_years):
                        yr_mask = (v_yr == yi) & valid_pnl
                        sums[combo_flat, yi] = pnl_arr[yr_mask].sum()
                        counts[combo_flat, yi] = yr_mask.sum()

    log.info(f"  Grid sweep complete.")
    return sums, counts


# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════

def walk_forward(sums, counts, years_list, min_trades=MIN_TRADES_TRAIN):
    """
    For each test year t (t >= 1):
      - Training = years [0..t-1]
      - Pick best combo from training (highest avg pips/trade with >= min_trades)
      - Apply blind to test year t
    Returns per-combo WF results and best-combo-per-year results.
    """
    n_years = len(years_list)
    n_combos = sums.shape[0]

    # Per-combo walk-forward (does this combo survive OOS?)
    combo_wf_pnl = np.zeros(n_combos, dtype=np.float64)
    combo_wf_n   = np.zeros(n_combos, dtype=np.int32)

    # Best-combo walk-forward (realistic: pick best from training each year)
    best_wf = []

    for ti in range(1, n_years):
        # Training: sum over years 0..ti-1
        train_sum = sums[:, :ti].sum(axis=1)
        train_cnt = counts[:, :ti].sum(axis=1)
        valid_train = train_cnt >= min_trades
        train_avg = np.where(valid_train, train_sum / np.maximum(train_cnt, 1), -1e10)

        # Per-combo WF: if profitable in training, count test PnL
        profitable = train_avg > 0
        combo_wf_pnl += np.where(profitable, sums[:, ti], 0.0)
        combo_wf_n   += np.where(profitable, counts[:, ti], 0).astype(np.int32)

        # Best-combo WF
        if not valid_train.any():
            continue
        best_ci = int(train_avg.argmax())
        best_train_avg = float(train_avg[best_ci])
        if best_train_avg <= 0:
            continue
        test_pnl = float(sums[best_ci, ti])
        test_n = int(counts[best_ci, ti])
        coords = _decode_combo(best_ci)
        best_wf.append({
            'test_year': years_list[ti],
            'combo_idx': best_ci,
            'label': _combo_label(*coords),
            'train_avg': best_train_avg,
            'train_n': int(train_cnt[best_ci]),
            'test_pnl': test_pnl,
            'test_n': test_n,
            'test_avg': test_pnl / max(test_n, 1),
        })

    # Per-combo WF avg
    combo_wf_avg = np.where(combo_wf_n > 0, combo_wf_pnl / combo_wf_n, -1e10)

    return combo_wf_pnl, combo_wf_n, combo_wf_avg, best_wf


# ══════════════════════════════════════════════════════════════════════
# COMPONENT CONTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def component_contribution(sums, counts, years_list):
    """
    For each filter dimension, compare average PnL when ON vs OFF.
    This shows which layers actually contribute edge.
    """
    n_combos = sums.shape[0]
    # Total PnL and trades per combo (across all years, excluding warmup year 0)
    total_pnl = sums[:, 1:].sum(axis=1)
    total_n   = counts[:, 1:].sum(axis=1)
    avg_pnl   = np.where(total_n >= MIN_TRADES_TRAIN, total_pnl / total_n, np.nan)

    contributions = {}

    for dim_name, dim_size, dim_labels in [
        ('D1 Filter', 3, ['off', 'ema(20,50)', 'ema(50,200)']),
        ('ADX Gate', 3, ['off', '≥20', '≥25']),
        ('Hurst Gate', 2, ['off', '≥0.53']),
        ('VoV Gate', 2, ['off', 'reject>P75']),
        ('RSI Div Block', 2, ['off', 'on']),
        ('Entry Method', 2, ['immediate', 'pullback']),
        ('Exit Method', 4, ['regime_flip', 'atr_2.5', 'atr_3.0', 'regime_tighten']),
        ('Partial Exit', 2, ['off', '50%@1R']),
    ]:
        dim_avgs = {}
        for combo_i in range(n_combos):
            coords = _decode_combo(combo_i)
            # Map dim name to coord index
            dim_map = {'D1 Filter': 0, 'ADX Gate': 1, 'Hurst Gate': 2,
                       'VoV Gate': 3, 'RSI Div Block': 4, 'Entry Method': 5,
                       'Exit Method': 6, 'Partial Exit': 7}
            val = coords[dim_map[dim_name]]
            if val not in dim_avgs:
                dim_avgs[val] = []
            if not np.isnan(avg_pnl[combo_i]):
                dim_avgs[val].append(avg_pnl[combo_i])

        results = []
        for vi in range(dim_size):
            vals = dim_avgs.get(vi, [])
            if vals:
                results.append((dim_labels[vi], np.mean(vals), len(vals)))
            else:
                results.append((dim_labels[vi], 0.0, 0))
        contributions[dim_name] = results

    return contributions


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS & REPORTING
# ══════════════════════════════════════════════════════════════════════

def _hdr(title, w=120):
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


def print_top_combos(sums, counts, years_list, n_top=20):
    """Print top N combos by OOS average pips/trade."""
    n_combos = sums.shape[0]
    # OOS = all years except year 0 (warmup)
    oos_pnl = sums[:, 1:].sum(axis=1)
    oos_n   = counts[:, 1:].sum(axis=1)
    oos_avg = np.where(oos_n >= MIN_TRADES_TRAIN, oos_pnl / oos_n, -1e10)

    ranked = np.argsort(-oos_avg)

    _hdr(f"TOP {n_top} COMBOS — by OOS avg pips/trade (year 0 = warmup)")
    print(f"\n  {'#':>3} {'Trades':>7} {'Total':>10} {'Avg':>8} {'WR':>6} {'PF':>6} "
          f"{'Config'}")
    print("  " + "-" * 115)

    for rank, ci in enumerate(ranked[:n_top]):
        if oos_avg[ci] <= -1e9:
            break
        coords = _decode_combo(ci)
        label = _combo_label(*coords)
        total = oos_pnl[ci]
        n = oos_n[ci]
        avg = total / max(n, 1)
        # Compute WR and PF from per-year data (approximate from total)
        print(f"  {rank+1:>3} {n:>7,} {total:>+10,.1f} {avg:>+8.2f} "
              f"{'—':>6} {'—':>6} {label}")

    return ranked[:n_top]


def print_yearly_breakdown(sums, counts, years_list, combo_idx, label):
    """Year-by-year breakdown for a specific combo."""
    print(f"\n  {label}")
    print(f"  {'Year':>6} {'Trades':>7} {'Total':>10} {'Avg':>8}")
    total_pnl = 0.0; total_n = 0
    for yi, yr in enumerate(years_list):
        t = sums[combo_idx, yi]
        n = counts[combo_idx, yi]
        a = t / max(n, 1) if n > 0 else 0
        marker = "+" if t > 0 else ("-" if t < 0 else " ")
        print(f"  {yr:>6} {n:>7,} {t:>+10,.1f} {a:>+8.2f}  {marker}")
        total_pnl += t; total_n += n
    print(f"  {'TOTAL':>6} {total_n:>7,} {total_pnl:>+10,.1f} "
          f"{total_pnl/max(total_n,1):>+8.2f}")


def print_component_analysis(contributions):
    _hdr("COMPONENT CONTRIBUTION ANALYSIS")
    print("\n  Which layers add edge? Average pips/trade across all combos where\n"
          "  that setting is used (OOS years only).\n")
    for dim_name, results in contributions.items():
        print(f"  {dim_name}:")
        best_val = max(r[1] for r in results)
        for label, avg, n_combos in results:
            marker = " ★" if avg == best_val and n_combos > 0 else ""
            print(f"    {label:<20} avg {avg:>+8.3f} pips/trade  "
                  f"({n_combos:,} combos){marker}")
        print()


def print_walk_forward_results(best_wf, years_list):
    _hdr("WALK-FORWARD VALIDATION — Best combo from training applied blind to test")
    if not best_wf:
        print("\n  No valid walk-forward folds.")
        return

    print(f"\n  {'Test Year':>10} {'Train Avg':>10} {'Test Trades':>12} "
          f"{'Test PnL':>10} {'Test Avg':>10} {'Result':>10}")
    print("  " + "-" * 70)
    total_pnl = 0.0; total_n = 0
    for fold in best_wf:
        result = "WIN" if fold['test_pnl'] > 0 else "LOSS"
        print(f"  {fold['test_year']:>10} {fold['train_avg']:>+10.2f} "
              f"{fold['test_n']:>12,} {fold['test_pnl']:>+10,.1f} "
              f"{fold['test_avg']:>+10.2f} {result:>10}")
        total_pnl += fold['test_pnl']
        total_n += fold['test_n']
    print(f"\n  WF Total: {total_pnl:>+,.1f} pips | {total_n:,} trades | "
          f"avg {total_pnl/max(total_n,1):>+.2f} pips/trade")

    wins = sum(1 for f in best_wf if f['test_pnl'] > 0)
    print(f"  WF Yearly Win Rate: {wins}/{len(best_wf)} "
          f"({100*wins/max(len(best_wf),1):.0f}%)")

    # Show which combos were selected
    print(f"\n  Selected combos per test year:")
    for fold in best_wf:
        print(f"    {fold['test_year']}: {fold['label']}")


def print_slippage_robustness(sums, counts, years_list, top_combos, n_show=5):
    _hdr(f"SLIPPAGE ROBUSTNESS — top {n_show} combos")
    rng = np.random.RandomState(RNG_SEED)

    for rank, ci in enumerate(top_combos[:n_show]):
        coords = _decode_combo(ci)
        label = _combo_label(*coords)
        oos_pnl = sums[ci, 1:].sum()
        oos_n = counts[ci, 1:].sum()
        if oos_n == 0:
            continue

        print(f"\n  #{rank+1}: {label}")
        print(f"  {'Slippage':>10} {'Total':>10} {'Avg':>8} {'Delta%':>8} {'OK?':>5}")
        base_avg = oos_pnl / oos_n

        for slip in SLIPPAGE_LEVELS:
            if slip == 0:
                print(f"  {'base':>10} {oos_pnl:>+10,.1f} {base_avg:>+8.2f} {'—':>8} "
                      f"{'YES' if oos_pnl > 0 else 'NO':>5}")
                continue
            # Simulate: each trade loses uniform(0, slip) additional pips
            degraded = oos_pnl - slip * 0.5 * oos_n  # expected value of uniform(0, slip)
            deg_avg = degraded / oos_n
            delta = (degraded - oos_pnl) / abs(oos_pnl) * 100 if oos_pnl != 0 else 0
            ok = "YES" if degraded > 0 else "NO"
            print(f"  {slip:>10.1f} {degraded:>+10,.1f} {deg_avg:>+8.2f} "
                  f"{delta:>+7.1f}% {ok:>5}")


def print_per_pair_breakdown(g, sums, counts, years_list, top_combos, pair_names, n_show=3):
    _hdr(f"PER-PAIR BREAKDOWN — top {n_show} combos")

    for rank, ci in enumerate(top_combos[:n_show]):
        coords = _decode_combo(ci)
        label = _combo_label(*coords)
        ent_i = coords[5]; ext_i = coords[6]; par_i = coords[7]

        print(f"\n  #{rank+1}: {label}")
        print(f"  {'Pair':<10} {'Trades':>7} {'Total':>10} {'Avg':>8}")
        print("  " + "-" * 40)

        # Reconstruct filter mask
        d1_i, adx_i, hur_i, vov_i, div_i = coords[:5]
        mask = np.ones(g['n_events'], dtype=np.bool_)
        if d1_i == 1: mask &= g['d1_20_50']
        elif d1_i == 2: mask &= g['d1_50_200']
        if ADX_OPTS[adx_i] > 0: mask &= g['adx'] >= ADX_OPTS[adx_i]
        if HURST_OPTS[hur_i] > 0: mask &= g['hurst'] >= HURST_OPTS[hur_i]
        mask &= g['vov_pctile'] < VOV_OPTS[vov_i][1]
        if div_i == 1: mask &= ~g['rsi_div_ag']
        mask &= g['entry_valid'][:, ent_i]
        # Exclude warmup year
        yr0 = min(g['year'])
        mask &= g['year'] > yr0

        pnl_col = g['pnl_partial' if par_i else 'pnl_full'][:, ent_i, ext_i]
        valid_pnl = ~np.isnan(pnl_col)
        mask &= valid_pnl

        n_pos = 0; n_neg = 0
        for pi, pname in enumerate(pair_names):
            pmask = mask & (g['pair_idx'] == pi)
            pnl = pnl_col[pmask]
            if len(pnl) == 0:
                continue
            t = pnl.sum(); n = len(pnl); a = t / n
            m = "+" if t > 0 else "-"
            if t > 0: n_pos += 1
            else: n_neg += 1
            print(f"  {pname:<10} {n:>7,} {t:>+10,.1f} {a:>+8.2f}  {m}")
        print(f"\n  Profitable pairs: {n_pos}/{n_pos+n_neg}")


def print_entry_vs_exit_matrix(sums, counts, years_list):
    """Show PnL by entry method × exit method (marginalised over filters)."""
    _hdr("ENTRY × EXIT METHOD MATRIX (avg pips/trade, OOS)")
    n_combos = sums.shape[0]
    oos_pnl = sums[:, 1:].sum(axis=1)
    oos_n   = counts[:, 1:].sum(axis=1)

    # Accumulate by (entry, exit) averaging over all filter combos
    matrix_sum = np.zeros((2, 4), dtype=np.float64)
    matrix_cnt = np.zeros((2, 4), dtype=np.float64)

    for ci in range(n_combos):
        coords = _decode_combo(ci)
        ent_i, ext_i = coords[5], coords[6]
        if oos_n[ci] >= MIN_TRADES_TRAIN:
            matrix_sum[ent_i, ext_i] += oos_pnl[ci] / oos_n[ci]
            matrix_cnt[ent_i, ext_i] += 1

    print(f"\n  {'':>12}", end='')
    for ex in EXIT_OPTS:
        print(f"  {ex:>16}", end='')
    print()
    for ent_i, ent_name in enumerate(['immediate', 'pullback']):
        print(f"  {ent_name:>12}", end='')
        for ext_i in range(4):
            if matrix_cnt[ent_i, ext_i] > 0:
                avg = matrix_sum[ent_i, ext_i] / matrix_cnt[ent_i, ext_i]
                print(f"  {avg:>+16.3f}", end='')
            else:
                print(f"  {'—':>16}", end='')
        print()


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Regime Deep Backtest — Multi-Layer Quant Strategy')
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir:
        for c in [r'C:\Forex_Projects\5_year_5s_data',
                  r'C:\Forex_Projects\Forex_bot_original',
                  r'C:\Forex_Projects\no_fucking_lookahead_bullshit_testing']:
            if os.path.isdir(c):
                data_dir = c; break
        if not data_dir:
            data_dir = os.getcwd()
    output_dir = args.output_dir or data_dir

    t_start = time_mod.time()

    _hdr("REGIME DEEP BACKTEST — Multi-Layer Quantitative Regime Strategy")
    print(f"  Grid: {N_TOTAL_COMBOS:,} combos "
          f"({N_FILTER_COMBOS} filter × {N_METHOD_COMBOS} method)")
    print(f"  Layers: D1 trend, ADX strength, Hurst persistence, VoV chaos,")
    print(f"          RSI divergence, pullback entry, ATR trail, partial exits,")
    print(f"          regime-tighten hybrid exit")
    print(f"  Base regime: ET={ENTRY_THRESH}, XT={EXIT_THRESH} (fixed from acid tests)")
    print(f"  Anti-cheat: bid/ask, no lookahead, walk-forward, all 19 pairs")
    print(f"  Data: {data_dir}")

    pairs = discover_pairs(data_dir)
    if not pairs:
        log.error(f"No parquet files in {data_dir}")
        sys.exit(1)

    pair_names = sorted(pairs.keys())
    log.info(f"  Pairs found: {len(pairs)} — {', '.join(pair_names)}")

    # ── Process all pairs ──
    all_events = []
    for pi, pname in enumerate(pair_names):
        t0 = time_mod.time()
        pip_m = 100.0 if 'JPY' in pname else 10000.0
        log.info(f"  Loading {pname} ...")
        df_5s = load_pair_data(pairs[pname])
        events = process_pair(df_5s, pname, pi, pip_m)
        all_events.extend(events)
        del df_5s
        log.info(f"    {pname}: {len(events):,} events ({time_mod.time()-t0:.1f}s)")

    log.info(f"  Total events: {len(all_events):,}")
    if not all_events:
        print("\n  NO EVENTS GENERATED — check data and regime parameters.")
        sys.exit(1)

    # ── Build global arrays ──
    g = build_global_arrays(all_events)
    years_list = sorted(set(g['year']))
    log.info(f"  Years: {years_list}")

    # ── Grid sweep ──
    sums, counts = run_grid_sweep(g, years_list)

    # ── Results ──
    top_combos = print_top_combos(sums, counts, years_list, n_top=25)

    # Yearly breakdown for top 5
    _hdr("YEARLY BREAKDOWN — top 5 combos")
    for rank, ci in enumerate(top_combos[:5]):
        coords = _decode_combo(ci)
        print_yearly_breakdown(sums, counts, years_list, ci,
                               f"#{rank+1}: {_combo_label(*coords)}")

    # Component contribution
    contributions = component_contribution(sums, counts, years_list)
    print_component_analysis(contributions)

    # Entry × Exit matrix
    print_entry_vs_exit_matrix(sums, counts, years_list)

    # Walk-forward
    combo_wf_pnl, combo_wf_n, combo_wf_avg, best_wf = walk_forward(sums, counts, years_list)
    print_walk_forward_results(best_wf, years_list)

    # Top WF combos
    _hdr("TOP 10 WALK-FORWARD COMBOS — by WF avg pips/trade")
    wf_ranked = np.argsort(-combo_wf_avg)
    print(f"\n  {'#':>3} {'WF Trades':>10} {'WF PnL':>10} {'WF Avg':>8} {'Config'}")
    print("  " + "-" * 100)
    for rank, ci in enumerate(wf_ranked[:10]):
        if combo_wf_avg[ci] <= -1e9:
            break
        coords = _decode_combo(ci)
        label = _combo_label(*coords)
        print(f"  {rank+1:>3} {combo_wf_n[ci]:>10,} {combo_wf_pnl[ci]:>+10,.1f} "
              f"{combo_wf_avg[ci]:>+8.2f} {label}")

    # Slippage
    print_slippage_robustness(sums, counts, years_list, top_combos)

    # Per-pair
    print_per_pair_breakdown(g, sums, counts, years_list, top_combos, pair_names)

    # ── Summary ──
    elapsed = time_mod.time() - t_start
    _hdr("FINAL SUMMARY")
    print(f"  Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Combos tested: {N_TOTAL_COMBOS:,}")
    print(f"  Events (all pairs): {len(all_events):,}")
    print(f"  Pairs: {len(pairs)}")

    oos_pnl = sums[:, 1:].sum(axis=1)
    oos_n = counts[:, 1:].sum(axis=1)
    oos_avg = np.where(oos_n >= MIN_TRADES_TRAIN, oos_pnl / oos_n, -1e10)
    n_profitable = (oos_avg > 0).sum()
    print(f"\n  Profitable combos (OOS): {n_profitable}/{N_TOTAL_COMBOS} "
          f"({100*n_profitable/N_TOTAL_COMBOS:.1f}%)")

    if best_wf:
        wf_total = sum(f['test_pnl'] for f in best_wf)
        wf_wins = sum(1 for f in best_wf if f['test_pnl'] > 0)
        print(f"\n  WALK-FORWARD VERDICT:")
        print(f"    Total PnL: {wf_total:>+,.1f} pips")
        print(f"    Yearly win rate: {wf_wins}/{len(best_wf)}")
        if wf_total > 0 and wf_wins >= len(best_wf) * 0.6:
            print(f"    ★ CONSISTENT EDGE DETECTED")
        elif wf_total > 0:
            print(f"    ◆ EDGE PRESENT BUT INCONSISTENT")
        else:
            print(f"    ✗ NO EDGE SURVIVED WALK-FORWARD")

    # Component summary
    print(f"\n  COMPONENT VERDICT (which layers help?):")
    for dim_name, results in contributions.items():
        if len(results) < 2:
            continue
        off_avg = results[0][1]
        best_on = max(r[1] for r in results[1:])
        delta = best_on - off_avg
        if delta > 0.1:
            verdict = f"HELPS (+{delta:.2f} pips/trade)"
        elif delta < -0.1:
            verdict = f"HURTS ({delta:.2f} pips/trade)"
        else:
            verdict = f"NEUTRAL ({delta:+.2f})"
        print(f"    {dim_name:<20} {verdict}")

    # ── Save results ──
    try:
        # Save top combos
        results_rows = []
        for ci in range(N_TOTAL_COMBOS):
            oos_p = oos_pnl[ci]; oos_c = oos_n[ci]
            if oos_c < MIN_TRADES_TRAIN:
                continue
            coords = _decode_combo(ci)
            results_rows.append({
                'd1_filter': D1_OPTS[coords[0]][0],
                'adx_gate': ADX_OPTS[coords[1]],
                'hurst_gate': HURST_OPTS[coords[2]],
                'vov_gate': VOV_OPTS[coords[3]][0],
                'rsi_div': DIV_OPTS[coords[4]],
                'entry': 'pullback' if coords[5] else 'immediate',
                'exit': EXIT_OPTS[coords[6]],
                'partial': PARTIAL_OPTS[coords[7]],
                'oos_trades': int(oos_c),
                'oos_total_pips': round(float(oos_p), 2),
                'oos_avg_pips': round(float(oos_p / oos_c), 4),
                'wf_trades': int(combo_wf_n[ci]),
                'wf_total_pips': round(float(combo_wf_pnl[ci]), 2),
                'wf_avg_pips': round(float(combo_wf_avg[ci]), 4) if combo_wf_n[ci] > 0 else 0,
            })
        if results_rows:
            rdf = pd.DataFrame(results_rows).sort_values('oos_avg_pips', ascending=False)
            out_path = os.path.join(output_dir, 'regime_deep_results.csv')
            rdf.to_csv(out_path, index=False)
            log.info(f"  Results saved: {out_path}")
    except Exception as e:
        log.warning(f"  Could not save results: {e}")

    print(f"\n{'='*120}")


if __name__ == '__main__':
    main()
