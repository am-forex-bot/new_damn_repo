#!/usr/bin/env python3
"""
REGIME ACID TEST — LIVE TRADING BOT (v2 — audit-fixed, spread bet ready)
=========================================================================
Trades EXACTLY like the backtest acid test. One fixed config. No optimisation.

Signal: MTF EMA bias → regime state machine
Entry:  Immediately on regime ON transition (0 bars delay)
Exit:   Immediately when regime OFF, OR 48h max hold
Sizing: % of account equity per trade, compounding as equity grows
Pairs:  All 19 (configurable)

ACCOUNT TYPES:
  SPREAD_BET=True  — units = GBP per pip (tax-free in UK). Default.
  SPREAD_BET=False — units = base currency (standard CFD).

AUDIT FIXES (v2):
  1. INCOMPLETE CANDLES — only use completed M5 bars (backtest never sees incomplete)
  2. MAX HOLD TIMER    — uses real UTC entry_time, survives restarts
  3. CANDLE HISTORY    — 5000 M5 bars (~17 days) for proper H4 EMA warmup
  4. TRANSITION GUARD  — regime state only evaluated on completed bars
  5. TOXIC HOUR FILTER — no entries 20:00-22:59 UTC (rollover, backtest-validated)
  6. SPREAD FILTER     — no entries when spread > 5 pips

CRITICAL FIXES (v3 — forensic audit):
  7. INCOMPLETE HIGHER-TF BARS — resample_tf now drops the last (incomplete) bar
     from M15, H1, H4 resampled data. Previously the H4 "close" updated every
     5 minutes (instead of every 4 hours), causing 75% of bias weight to flicker.
     THIS WAS THE PRIMARY CAUSE OF REGIME WHIPSAWING AND RAPID-FIRE LOSSES.
  8. TOXIC HOUR TRANSITION LOSS — prev_state is no longer updated when a
     transition is rejected for retry-able reasons (toxic hours, spread too wide,
     max concurrent). Previously, transitions during toxic hours were permanently
     lost because prev_state was updated before the entry check.
  9. TALIB EMA MATCH — uses talib.EMA if available (SMA-seeded, same as backtest).
     Falls back to numpy EMA if talib not installed. Eliminates any EMA divergence
     between bot and backtest when talib is present.
 10. MAX HOLD BAR COUNT — uses M5 bar count (576 = 48h market time), not wall
     clock. Weekends are automatically skipped (no M5 bars). Exactly matches
     backtest's TIMED_EXIT=576. Bar count persists in state.json for restarts.
 11. REAL M1 DATA — fetches actual M1 candles from OANDA instead of faking M1
     with M5 bars. Backtest builds real M1 from 5s tick data; this is the
     closest live equivalent. Eliminates M5 double-weighting.
 12. PERSISTENT CANDLE CACHE — M1 and M5 data cached to disk as pickle files.
     On restart, loads from cache and fetches only new bars since last timestamp.
     Dramatically reduces API load and ensures proper EMA warmup from first cycle.
 13. M1 FULL COVERAGE — fetches ~25000 M1 bars (~17 days) via paginated API
     calls to match the M5 data window. Previously only 5000 M1 bars (~3.5 days),
     meaning M1 signal was stale for the first ~13 days of the bias calculation.
     Now M1 covers the same time span as M5, eliminating any clipping artefacts.

Run:    python regime_acid_test_bot_v2.py
Stop:   Ctrl+C (closes all positions gracefully)
"""

import os, sys, time, logging, json, signal as sig_module
from logging.handlers import RotatingFileHandler
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import requests

# FIX #9: Use talib EMA if available — matches backtest exactly.
# talib.EMA uses SMA for the first `span` values, then switches to EMA.
# The numpy fallback uses first-value seeding. With 5000 bars of warmup
# the difference is negligible, but using the same implementation removes
# any possibility of signal divergence.
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

class Config:
    # ── OANDA ──
    OANDA_ACCOUNT_ID  = os.getenv('OANDA_ACCOUNT_ID', '')
    OANDA_ACCESS_TOKEN = os.getenv('OANDA_ACCESS_TOKEN', '')
    OANDA_ENVIRONMENT  = os.getenv('OANDA_ENV', 'practice')

    if OANDA_ENVIRONMENT == 'practice':
        OANDA_API_URL = 'https://api-fxpractice.oanda.com'
    else:
        OANDA_API_URL = 'https://api-fxtrade.oanda.com'

    # ── REGIME SIGNAL (FIXED — matches backtest exactly) ──
    ENTRY_THRESHOLD = 0.4
    EXIT_THRESHOLD  = 0.2
    MAX_HOLD_BARS   = 576      # FIX #10: 576 M5 bars = 48h market time (matches backtest TIMED_EXIT)
    MTF_WEIGHTS     = {'M1': 0.05, 'M5': 0.20, 'M15': 0.30, 'H1': 0.25, 'H4': 0.20}

    # ── ACCOUNT TYPE ──
    # None  = auto-detect from OANDA instrument metadata at startup
    # True  = force spread betting (units = GBP per pip, tax-free UK)
    # False = force CFD (units = base currency)
    SPREAD_BET = None  # auto-detect

    # ── POSITION SIZING ──
    EQUITY_PER_TRADE_PCT  = 0.02   # 2% of equity per position
    MAX_CONCURRENT        = 19     # all 19 pairs
    MAX_SPREAD_PIPS       = 5.0    # reject entries above this spread
    TOXIC_HOURS_UTC       = [20, 21, 22]  # reject entries during rollover (backtest-validated)
    MIN_UNITS             = 0.01   # OANDA spread bet minimum: 1p per pip

    # ── PAIRS ──
    PAIRS = [
        'EUR_USD', 'GBP_USD', 'USD_JPY', 'EUR_JPY', 'GBP_JPY',
        'AUD_USD', 'NZD_USD', 'USD_CAD', 'USD_CHF', 'EUR_GBP',
        'EUR_AUD', 'GBP_AUD', 'AUD_JPY', 'CAD_JPY', 'CHF_JPY',
        'NZD_JPY', 'AUD_CAD', 'NZD_CAD', 'AUD_NZD',
    ]

    # ── TIMING ──
    CHECK_INTERVAL_SECS = 10   # poll for new M5 bar every 10s
    # FIX #3: 5000 M5 bars = ~17 days. Ensures H4 EMA(21) has >400 H4 bars
    # for full convergence. Backtest had 5+ years. 17 days is the minimum
    # for H4 EMA to be within 0.1% of the fully-warmed value.
    CANDLE_HISTORY      = 5000  # M5 bars to fetch (OANDA max per request)
    M1_CANDLE_HISTORY   = 25000 # FIX #11+13: Real M1 bars covering same ~17 days as M5

    # ── DATA CACHE ──
    # FIX #12: Persistent candle cache survives restarts. On startup, loads
    # from disk and fetches only new bars since last cached timestamp.
    # Dramatically reduces API load (delta fetch vs 5000 bars per pair per cycle).
    CACHE_DIR = 'acid_test_cache'

    # ── LOGGING ──
    LOG_DIR  = 'acid_test_logs'
    LOG_FILE = 'acid_test_bot.log'
    JOURNAL_FILE = 'acid_test_journal.log'  # detailed signal decisions
    TRADE_CSV = 'acid_test_live_trades.csv'
    STATE_FILE = 'acid_test_state.json'  # persist trade state for restart recovery

    # Log rotation — prevents disk bloat
    LOG_MAX_BYTES    = 10 * 1024 * 1024  # 10 MB per file
    LOG_BACKUP_COUNT = 5                  # keep 5 rotated files (50 MB total max)
    JOURNAL_MAX_BYTES    = 20 * 1024 * 1024  # 20 MB per file (journal is verbose)
    JOURNAL_BACKUP_COUNT = 3                  # keep 3 rotated files (60 MB total max)


# ══════════════════════════════════════════════════════════════════════
# OANDA CLIENT (minimal, self-contained)
# ══════════════════════════════════════════════════════════════════════

class OandaClient:
    def __init__(self):
        self.base = Config.OANDA_API_URL
        self.acct = Config.OANDA_ACCOUNT_ID
        self.sess = requests.Session()
        self.sess.headers.update({
            'Authorization': f'Bearer {Config.OANDA_ACCESS_TOKEN}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339',
        })

    def _get(self, path, params=None, retries=3):
        for attempt in range(retries):
            try:
                r = self.sess.get(f'{self.base}{path}', params=params, timeout=30)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                logging.error(f'GET {path} → {r.status_code}: {r.text[:200]}')
            except Exception as e:
                logging.error(f'GET {path} error: {e}')
            time.sleep(1)
        return None

    def _post(self, path, data, retries=3):
        for attempt in range(retries):
            try:
                r = self.sess.post(f'{self.base}{path}', json=data, timeout=30)
                if r.status_code in (200, 201):
                    return r.json()
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                logging.error(f'POST {path} → {r.status_code}: {r.text[:200]}')
            except Exception as e:
                logging.error(f'POST {path} error: {e}')
            time.sleep(1)
        return None

    def _put(self, path, data, retries=3):
        for attempt in range(retries):
            try:
                r = self.sess.put(f'{self.base}{path}', json=data, timeout=30)
                if r.status_code in (200, 201):
                    return r.json()
                logging.error(f'PUT {path} → {r.status_code}: {r.text[:200]}')
            except Exception as e:
                logging.error(f'PUT {path} error: {e}')
            time.sleep(1)
        return None

    def get_account(self):
        data = self._get(f'/v3/accounts/{self.acct}/summary')
        return data.get('account', {}) if data else {}

    def fetch_candles(self, pair, granularity, count, price='MBA'):
        data = self._get(f'/v3/instruments/{pair}/candles', {
            'granularity': granularity, 'count': min(count, 5000), 'price': price,
        })
        return data.get('candles', []) if data else []

    def fetch_candles_since(self, pair, granularity, since_time, price='MBA', max_count=500):
        """Fetch candles from a specific time onwards (for incremental cache updates)."""
        from_str = since_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
        data = self._get(f'/v3/instruments/{pair}/candles', {
            'granularity': granularity, 'from': from_str,
            'count': min(max_count, 5000), 'price': price,
        })
        return data.get('candles', []) if data else []

    def fetch_candles_paginated(self, pair, granularity, total_count, price='MBA'):
        """
        FIX #13: Fetch more than 5000 candles by paginating forward.

        OANDA limits each request to 5000 candles. For M1 data covering ~17 days
        (~25000 bars), this paginates multiple requests forward from a calculated
        start time, using 'from' + 'count' per page (OANDA rejects 'from'+'to'
        if the implied count exceeds 5000).

        Used only for cold-start (no cached data). Once the cache is populated,
        incremental fetch_candles_since is used instead.
        """
        if total_count <= 5000:
            return self.fetch_candles(pair, granularity, total_count, price)

        # Map granularity to minutes per bar
        gran_minutes = {'S5': 5/60, 'S10': 10/60, 'S15': 15/60, 'S30': 30/60,
                        'M1': 1, 'M2': 2, 'M4': 4, 'M5': 5, 'M10': 10, 'M15': 15,
                        'M30': 30, 'H1': 60, 'H2': 120, 'H3': 180, 'H4': 240,
                        'H6': 360, 'H8': 480, 'H12': 720, 'D': 1440, 'W': 10080}
        bar_mins = gran_minutes.get(granularity, 5)

        # Start far enough back (1.5x buffer for weekends/gaps)
        start_time = datetime.now(timezone.utc) - timedelta(minutes=bar_mins * total_count * 1.5)
        all_candles = []
        page = 0

        while len(all_candles) < total_count:
            from_str = start_time.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            data = self._get(f'/v3/instruments/{pair}/candles', {
                'granularity': granularity, 'from': from_str,
                'count': 5000, 'price': price,
            })
            candles = data.get('candles', []) if data else []

            if not candles:
                break

            all_candles.extend(candles)
            page += 1

            # Move start_time forward to after the last candle
            last_time = pd.Timestamp(candles[-1]['time']).to_pydatetime()
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=timezone.utc)
            start_time = last_time + timedelta(seconds=1)

            logging.info(f'    {pair}/{granularity}: page {page} fetched {len(candles)} bars '
                         f'(total so far: {len(all_candles)})')
            time.sleep(0.3)  # rate limit between pages

            if len(candles) < 5000:
                break  # reached current time

        return all_candles

    def get_price(self, pair):
        data = self._get(f'/v3/accounts/{self.acct}/pricing',
                         {'instruments': pair})
        if data and data.get('prices'):
            p = data['prices'][0]
            return float(p['bids'][0]['price']), float(p['asks'][0]['price'])
        return None, None

    def place_market(self, pair, units, comment=''):
        # Spread bet: decimal GBP/pip (e.g. "0.01"); CFD: integer base units (e.g. "120")
        if Config.SPREAD_BET:
            units_str = f'{units:.2f}'
        else:
            units_str = str(int(units))
        order = {
            'order': {
                'type': 'MARKET',
                'instrument': pair,
                'units': units_str,
                'timeInForce': 'FOK',
                'positionFill': 'DEFAULT',
            }
        }
        if comment:
            order['order']['clientExtensions'] = {'comment': comment[:128]}
        result = self._post(f'/v3/accounts/{self.acct}/orders', order)
        if result:
            fill = result.get('orderFillTransaction')
            if fill:
                return fill
            reject = result.get('orderRejectTransaction')
            if reject:
                logging.error(f'REJECTED {pair}: {reject.get("rejectReason")}')
        return None

    def get_instrument_info(self, pair):
        """Fetch instrument metadata (tradeUnitsPrecision, minimumTradeSize, etc.)."""
        data = self._get(f'/v3/accounts/{self.acct}/instruments',
                         {'instruments': pair})
        if data and data.get('instruments'):
            return data['instruments'][0]
        return None

    def get_open_trades(self):
        data = self._get(f'/v3/accounts/{self.acct}/openTrades')
        return data.get('trades', []) if data else []

    def close_trade(self, trade_id):
        result = self._put(f'/v3/accounts/{self.acct}/trades/{trade_id}/close', {})
        if result:
            return result.get('orderFillTransaction')
        return None


# ══════════════════════════════════════════════════════════════════════
# DATA CACHE — persistent M1/M5 candle storage
# ══════════════════════════════════════════════════════════════════════

class DataCache:
    """
    FIX #12: Persistent candle cache for all pairs.

    Stores M1 and M5 data as pickle files on disk. On startup, loads
    from disk and fetches only new bars since the last cached timestamp.
    On each cycle, appends new bars and trims old data.

    Benefits:
      1. Real M1 data (not faked from M5) — FIX #11
      2. Restart recovery without re-fetching entire history
      3. Reduced API load (delta fetch vs 5000 bars per pair per cycle)
      4. Proper EMA warmup from the first cycle
    """

    TIMEFRAMES = ['M1', 'M5']
    MAX_BARS = {'M1': 25000, 'M5': 5000}  # M1: ~17 days to match M5 coverage

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.data = {}  # {pair: {'M1': DataFrame, 'M5': DataFrame}}

    def _path(self, pair, tf):
        return os.path.join(self.cache_dir, f'{pair}_{tf}.pkl')

    def load_all(self, pairs):
        """Load all cached data from disk. Returns count of files loaded."""
        loaded = 0
        for pair in pairs:
            self.data[pair] = {}
            for tf in self.TIMEFRAMES:
                path = self._path(pair, tf)
                if os.path.exists(path):
                    try:
                        df = pd.read_pickle(path)
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            self.data[pair][tf] = df
                            loaded += 1
                        else:
                            self.data[pair][tf] = pd.DataFrame()
                    except Exception as e:
                        logging.warning(f'  Cache load failed {pair}/{tf}: {e}')
                        self.data[pair][tf] = pd.DataFrame()
                else:
                    self.data[pair][tf] = pd.DataFrame()
        return loaded

    def save(self, pair, tf):
        """Save a pair's timeframe data to disk."""
        df = self.data.get(pair, {}).get(tf)
        if df is not None and len(df) > 0:
            df.to_pickle(self._path(pair, tf))

    def update(self, pair, tf, new_df):
        """Append new candles, deduplicate, trim old data, save to disk."""
        if new_df is None or len(new_df) == 0:
            return

        existing = self.data.get(pair, {}).get(tf, pd.DataFrame())
        if len(existing) > 0:
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
        else:
            combined = new_df.sort_index()

        max_bars = self.MAX_BARS.get(tf, 5000)
        if len(combined) > max_bars:
            combined = combined.iloc[-max_bars:]

        self.data.setdefault(pair, {})[tf] = combined
        self.save(pair, tf)

    def get(self, pair, tf):
        """Get cached DataFrame for a pair/timeframe."""
        return self.data.get(pair, {}).get(tf, pd.DataFrame())

    def last_time(self, pair, tf):
        """Get the last cached timestamp for a pair/timeframe."""
        df = self.get(pair, tf)
        if len(df) > 0:
            return df.index[-1]
        return None

    def bar_count(self, pair, tf):
        """Get number of cached bars."""
        return len(self.get(pair, tf))

    def summary(self):
        """Return summary string of cache contents."""
        lines = []
        for pair in sorted(self.data.keys()):
            parts = []
            for tf in self.TIMEFRAMES:
                n = self.bar_count(pair, tf)
                last = self.last_time(pair, tf)
                last_str = last.strftime('%Y-%m-%d %H:%M') if last else 'empty'
                parts.append(f'{tf}={n} bars (→{last_str})')
            lines.append(f'    {pair}: {" | ".join(parts)}')
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════
# SIGNAL ENGINE (identical maths to backtest)
# ══════════════════════════════════════════════════════════════════════

def _ema_np(arr, span):
    """Numpy EMA fallback — first-value seeded."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def ema(arr, span):
    """
    EMA matching backtest exactly.

    FIX #9: Use talib.EMA if available (SMA-seeded, same as backtest).
    Falls back to numpy implementation if talib not installed.
    """
    if HAS_TALIB:
        return talib.EMA(arr, timeperiod=span)
    return _ema_np(arr, span)


def candles_to_df(candles):
    """
    Convert OANDA candle list to DataFrame.

    FIX #1: ALWAYS skip incomplete candles. The backtest NEVER processes
    incomplete bars — it uses finalised historical data. Including an
    incomplete bar causes signal flicker as price updates within the bar,
    which triggers false regime transitions.
    """
    rows = []
    for c in candles:
        # FIX #1: ALWAYS skip incomplete candles, no exceptions
        if not c.get('complete', False):
            continue
        mid = c.get('mid', {})
        bid = c.get('bid', {})
        ask = c.get('ask', {})
        rows.append({
            'time': pd.Timestamp(c['time']),
            'open': float(mid.get('o', 0)),
            'high': float(mid.get('h', 0)),
            'low': float(mid.get('l', 0)),
            'close': float(mid.get('c', 0)),
            'bid_close': float(bid.get('c', mid.get('c', 0))),
            'ask_close': float(ask.get('c', mid.get('c', 0))),
            'volume': int(c.get('volume', 0)),
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.set_index('time').sort_index()
    return df


def resample_tf(m5_df, rule):
    """
    Resample M5 to higher timeframe, dropping the last (incomplete) bar.

    FIX #7: The last bar of each resampled timeframe is almost always incomplete.
    At 02:15 UTC, the H4 bar (00:00-04:00) only has 27 of 48 M5 bars.
    Using that partial bar's close as if it were a real H4 close causes the
    H4 EMA to update every 5 minutes instead of every 4 hours, making the
    signal flicker and triggering false regime transitions.

    The backtest never has this problem because all data is historical (complete).
    Dropping the last bar ensures we only use fully-formed higher-TF bars,
    exactly matching the backtest behaviour.
    """
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
           'bid_close': 'last', 'ask_close': 'last', 'volume': 'sum'}
    df = m5_df.resample(rule).agg(agg).dropna(subset=['close'])
    # Drop the last bar — it's almost always incomplete (partially filled
    # from the current higher-TF period). Only fully-formed bars should
    # feed into EMA calculations.
    if len(df) > 1:
        df = df.iloc[:-1]
    return df


def compute_mtf_bias(m5_df, m1_df=None):
    """
    Compute MTF bias from M5 data and real M1 data — matches backtest.
    Resamples M5 to M15, H1, H4, computes EMAs, derives bias.

    FIX #11: Uses real M1 data from OANDA when available (cached separately).
    Falls back to M5 proxy only if M1 data is insufficient.
    The backtest builds M1 from 5-second tick data — using real M1 candles
    from OANDA is the closest match possible in a live environment.
    """
    # Use real M1 if available and has enough bars for EMA(21), else fallback to M5
    if m1_df is not None and len(m1_df) >= 22:
        m1_source = m1_df
    else:
        m1_source = m5_df  # fallback only if M1 data insufficient

    tfs = {
        'M5': m5_df,
        'M1': m1_source,
        'M15': resample_tf(m5_df, '15min'),
        'H1': resample_tf(m5_df, '1h'),
        'H4': resample_tf(m5_df, '4h'),
    }

    n = len(m5_df)
    bias = np.zeros(n, dtype=np.float64)
    m5_ns = m5_df.index.asi8

    for tf_name, weight in Config.MTF_WEIGHTS.items():
        df = tfs.get(tf_name)
        if df is None or len(df) < 22:
            continue
        c = df['close'].values.astype(np.float64)
        e9 = ema(c, 9)
        e21 = ema(c, 21)
        sig = np.where((e9 > e21) & (c > e9), 1.0,
                       np.where((e9 < e21) & (c < e9), -1.0, 0.0))
        tt = df.index.asi8
        idx = np.clip(np.searchsorted(tt, m5_ns, side='right') - 1, 0, len(sig) - 1)
        bias += sig[idx] * weight

    total_w = sum(Config.MTF_WEIGHTS.values())
    if total_w > 0 and total_w != 1.0:
        bias /= total_w
    return np.clip(bias, -1.0, 1.0)


def regime_state_machine(bias, et, xt):
    """Regime hysteresis — matches backtest exactly."""
    n = len(bias)
    state = np.zeros(n, dtype=np.int32)
    cur = 0
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


def pip_mult(pair):
    return 100.0 if 'JPY' in pair else 10000.0


# ══════════════════════════════════════════════════════════════════════
# POSITION SIZER
# ══════════════════════════════════════════════════════════════════════

def calc_units(pair, direction, equity, bid, ask):
    """
    Calculate position size.

    SPREAD BET MODE (SPREAD_BET=True):
      units = GBP per pip. Formula targets 2% margin per trade.
      margin = stake × (mid_price / pip_size) / leverage
      ∴ stake = equity × risk% × leverage × pip_size / mid_price

      £200 account, EUR/USD at 1.08:
        stake = 200 × 0.02 × 30 × 0.0001 / 1.08 = £0.011/pip → rounds to £0.01/pip
        margin = 0.01 × (1.08 / 0.0001) / 30 = 0.01 × 10,800 / 30 = £3.60
        19 concurrent × £3.60 = ~£68 total margin (34% of £200)

      USD/JPY at 150.0:
        stake = 200 × 0.02 × 30 × 0.01 / 150 = £0.008/pip → rounds to £0.01/pip
        margin = 0.01 × (150 / 0.01) / 30 = 0.01 × 15,000 / 30 = £5.00

      EUR/GBP at 0.85:
        stake = 200 × 0.02 × 30 × 0.0001 / 0.85 = £0.014/pip → rounds to £0.01/pip
        margin = 0.01 × (0.85 / 0.0001) / 30 = 0.01 × 8,500 / 30 = £2.83

    CFD MODE (SPREAD_BET=False):
      units = base currency. 200 × 0.02 × 30 = 120 units.

    As equity grows, stake grows proportionally → compounding.
    """
    leverage = 30.0  # OANDA ESMA retail max for major pairs

    if Config.SPREAD_BET:
        # Spread bet: units = GBP per pip
        mid = (bid + ask) / 2.0
        pip_sz = 0.01 if 'JPY' in pair else 0.0001
        stake = equity * Config.EQUITY_PER_TRADE_PCT * leverage * pip_sz / mid
        stake = round(stake, 2)  # OANDA spread bet precision: 2 decimal places
        if stake < Config.MIN_UNITS:
            stake = Config.MIN_UNITS
        return stake * direction
    else:
        # CFD: units = base currency (integer)
        units = int(equity * Config.EQUITY_PER_TRADE_PCT * leverage)
        if units < Config.MIN_UNITS:
            units = int(Config.MIN_UNITS)
        return units * direction


# ══════════════════════════════════════════════════════════════════════
# TRADE TRACKER — with persistent state for restart recovery
# ══════════════════════════════════════════════════════════════════════

class TradeTracker:
    """Track open positions and enforce regime-based exits."""

    def __init__(self):
        self.open_trades = {}  # pair → {trade_id, direction, entry_time_utc, entry_price, bars_held}

    def has_position(self, pair):
        return pair in self.open_trades

    def add_trade(self, pair, trade_id, direction, entry_price, entry_time_utc):
        """
        Store trade with both UTC entry time (for logging/audit) and bar counter
        (for max hold matching backtest's TIMED_EXIT=576).
        """
        self.open_trades[pair] = {
            'trade_id': trade_id,
            'direction': direction,
            'entry_time_utc': entry_time_utc.isoformat() if isinstance(entry_time_utc, datetime) else entry_time_utc,
            'entry_price': entry_price,
            'bars_held': 0,  # FIX #10: Count M5 bars for max hold (matches backtest)
        }
        self._save_state()

    def remove_trade(self, pair):
        result = self.open_trades.pop(pair, None)
        self._save_state()
        return result

    def increment_bars_held(self, pair):
        """
        FIX #10: Increment bar counter for a trade (called on each new M5 bar).
        This counts actual market-time M5 bars, exactly like the backtest's
        TIMED_EXIT=576. Weekends are automatically skipped because no M5 bars
        are generated when the market is closed.
        """
        if pair in self.open_trades:
            self.open_trades[pair]['bars_held'] = self.open_trades[pair].get('bars_held', 0) + 1
            # Don't save on every bar increment — save is called at end of cycle
            # to avoid excessive disk I/O during the scan phase.

    def save_state_if_changed(self):
        """Save state to disk (call once per cycle, not per bar increment)."""
        self._save_state()

    def get_entry_time(self, pair):
        """Return entry time as datetime for a tracked trade."""
        info = self.open_trades.get(pair)
        if not info:
            return None
        et = info['entry_time_utc']
        if isinstance(et, str):
            return datetime.fromisoformat(et)
        return et

    def get_trades_to_close(self, pair_states):
        """
        FIX #10: Max hold uses M5 bar count, not wall clock.

        Return list of (pair, trade_info, reason) for trades that should close.

        576 M5 bars = 48 hours of market time (matches backtest TIMED_EXIT=576).
        This automatically skips weekends — no M5 bars are generated when
        the market is closed, so weekend time doesn't count towards max hold.
        This matches the backtest exactly, where TIMED_EXIT=576 bars also
        implicitly skips weekends because there's no weekend data.

        Survives restarts: bars_held is persisted in state.json.
        """
        to_close = []
        now = datetime.now(timezone.utc)

        for pair, info in list(self.open_trades.items()):
            bars_held = info.get('bars_held', 0)

            # For logging: also compute wall-clock elapsed time
            entry_time = self.get_entry_time(pair)
            if entry_time and entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            elapsed_hours = (now - entry_time).total_seconds() / 3600.0 if entry_time else 0

            # Max hold check (bar count — matches backtest)
            if bars_held >= Config.MAX_HOLD_BARS:
                to_close.append((pair, info,
                    f'MAX_HOLD ({bars_held} bars / ~{elapsed_hours:.1f}h wall)'))
                continue

            # Regime off check
            state = pair_states.get(pair, 0)
            if state != info['direction']:
                to_close.append((pair, info,
                    f'REGIME_OFF (was {info["direction"]}, now {state}, '
                    f'{bars_held} bars / ~{elapsed_hours:.1f}h wall)'))

        return to_close

    def _save_state(self):
        """Persist trade state to disk for restart recovery."""
        try:
            state_path = os.path.join(Config.LOG_DIR, Config.STATE_FILE)
            with open(state_path, 'w') as f:
                json.dump(self.open_trades, f, indent=2)
        except Exception as e:
            logging.error(f'Failed to save state: {e}')

    def _load_state(self):
        """Load trade state from disk after restart."""
        try:
            state_path = os.path.join(Config.LOG_DIR, Config.STATE_FILE)
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    self.open_trades = json.load(f)
                logging.info(f'  Loaded {len(self.open_trades)} trades from state file')
                return True
        except Exception as e:
            logging.error(f'Failed to load state: {e}')
        return False


# ══════════════════════════════════════════════════════════════════════
# TRADE LOGGER
# ══════════════════════════════════════════════════════════════════════

class TradeLogger:
    def __init__(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        self.csv_path = os.path.join(Config.LOG_DIR, Config.TRADE_CSV)
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write('timestamp,pair,direction,units,entry_price,exit_price,'
                        'pnl_pips,pnl_gbp,hold_hours,exit_reason,'
                        'equity_before,equity_after\n')

    def log_trade(self, pair, direction, units, entry_price, exit_price,
                  entry_time_utc, exit_reason, equity_before, equity_after):
        pm = pip_mult(pair)
        pnl_pips = direction * (exit_price - entry_price) * pm
        pnl_gbp = equity_after - equity_before  # actual realised

        # Calculate actual hold time from real timestamps
        if isinstance(entry_time_utc, str):
            entry_time_utc = datetime.fromisoformat(entry_time_utc)
        if entry_time_utc.tzinfo is None:
            entry_time_utc = entry_time_utc.replace(tzinfo=timezone.utc)
        hold_hours = (datetime.now(timezone.utc) - entry_time_utc).total_seconds() / 3600.0

        with open(self.csv_path, 'a') as f:
            f.write(f'{datetime.now(timezone.utc).isoformat()},{pair},{direction},'
                    f'{units},{entry_price:.6f},{exit_price:.6f},'
                    f'{pnl_pips:.2f},{pnl_gbp:.2f},{hold_hours:.2f},'
                    f'{exit_reason},{equity_before:.2f},{equity_after:.2f}\n')

        logging.info(f'  CLOSED {pair} | {pnl_pips:+.1f} pips | £{pnl_gbp:+.2f} | '
                     f'{hold_hours:.1f}h | {exit_reason}')


# ══════════════════════════════════════════════════════════════════════
# MAIN BOT
# ══════════════════════════════════════════════════════════════════════

class AcidTestBot:
    def __init__(self):
        self.client = OandaClient()
        self.tracker = TradeTracker()
        self.trade_log = TradeLogger()
        self.cache = DataCache(Config.CACHE_DIR)  # FIX #12: persistent candle cache
        self.last_m5_time = {}  # pair → last processed M5 bar time
        self.prev_state = {}    # pair → previous regime state
        self.running = True
        self.stats = {'entries': 0, 'exits': 0, 'errors': 0}

    def setup_logging(self):
        os.makedirs(Config.LOG_DIR, exist_ok=True)

        # ── Main log: INFO level, rotating, console + file ──
        log_path = os.path.join(Config.LOG_DIR, Config.LOG_FILE)
        fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)  # allow all levels, handlers filter

        file_handler = RotatingFileHandler(
            log_path, maxBytes=Config.LOG_MAX_BYTES,
            backupCount=Config.LOG_BACKUP_COUNT)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(fmt)

        root.addHandler(file_handler)
        root.addHandler(console_handler)

        # ── Journal log: detailed signal decisions, separate file ──
        journal_path = os.path.join(Config.LOG_DIR, Config.JOURNAL_FILE)
        journal_fmt = logging.Formatter(
            '%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        journal_handler = RotatingFileHandler(
            journal_path, maxBytes=Config.JOURNAL_MAX_BYTES,
            backupCount=Config.JOURNAL_BACKUP_COUNT)
        journal_handler.setLevel(logging.DEBUG)
        journal_handler.setFormatter(journal_fmt)

        self.journal = logging.getLogger('journal')
        self.journal.addHandler(journal_handler)
        self.journal.setLevel(logging.DEBUG)
        self.journal.propagate = False  # don't duplicate to main log

    def get_equity(self):
        acct = self.client.get_account()
        if acct:
            return float(acct.get('NAV', acct.get('balance', 0)))
        return 0.0

    def update_cache_for_pair(self, pair):
        """
        FIX #12: Incrementally update the candle cache for a pair.

        If cache has recent data, fetches only new bars since last cached time.
        If cache is empty (first run or corrupted), fetches full history.
        This replaces the old approach of fetching 5000 bars per pair per cycle.

        FIX #13: For M1 cold starts, uses paginated fetch to get ~17 days of M1
        data (matching M5 coverage). OANDA limits requests to 5000 candles, so
        25000 M1 bars requires 5 paginated requests.
        """
        for tf, history_count in [('M5', Config.CANDLE_HISTORY),
                                   ('M1', Config.M1_CANDLE_HISTORY)]:
            last = self.cache.last_time(pair, tf)
            if last is not None:
                # Incremental: fetch only new bars since last cached time
                candles = self.client.fetch_candles_since(pair, tf, last)
            elif history_count > 5000:
                # Cold start with pagination (M1 needs >5000 bars)
                candles = self.client.fetch_candles_paginated(pair, tf, history_count)
            else:
                # Cold start: single request (M5 fits in one 5000-bar request)
                candles = self.client.fetch_candles(pair, tf, history_count)

            if candles:
                new_df = candles_to_df(candles)  # FIX #1: incomplete candles filtered
                self.cache.update(pair, tf, new_df)

    def get_regime_state(self, pair):
        """
        Compute regime state from cached M5/M1 data.

        FIX #1:  Only uses completed candles.
        FIX #7:  Higher-TF bars drop incomplete last bar.
        FIX #9:  Uses talib EMA when available.
        FIX #11: Uses real M1 data (not M5 proxy).
        FIX #12: Uses persistent cache (incremental updates).

        Returns: (state, bid_close, ask_close, bar_time, bias_value, spread_pips)
                 or (0, None, None, None, None, None) on error.
        """
        # Update cache with any new bars
        self.update_cache_for_pair(pair)

        m5 = self.cache.get(pair, 'M5')
        m1 = self.cache.get(pair, 'M1')

        if len(m5) < 100:
            return 0, None, None, None, None, None

        bias = compute_mtf_bias(m5, m1)
        state = regime_state_machine(bias, Config.ENTRY_THRESHOLD, Config.EXIT_THRESHOLD)

        current_state = int(state[-1])
        current_bias = float(bias[-1])
        bid_c = float(m5['bid_close'].iloc[-1])
        ask_c = float(m5['ask_close'].iloc[-1])
        spread = (ask_c - bid_c) * pip_mult(pair)
        last_time = m5.index[-1]

        return current_state, bid_c, ask_c, last_time, current_bias, spread

    def check_spread(self, pair, bid, ask):
        """Check spread is acceptable."""
        spread_pips = (ask - bid) * pip_mult(pair)
        if spread_pips > Config.MAX_SPREAD_PIPS:
            return False, spread_pips
        return True, spread_pips

    def try_entry(self, pair, direction, bid, ask):
        """Attempt to enter a trade."""
        if self.tracker.has_position(pair):
            return  # already in a trade for this pair

        # Check toxic hours (rollover period — backtest-validated)
        now_hour = datetime.now(timezone.utc).hour
        if now_hour in Config.TOXIC_HOURS_UTC:
            logging.debug(f'  {pair}: toxic hour {now_hour}:00 UTC — skipping entry')
            return

        # Check spread using LIVE quote (not stale M5 bar close)
        # FIX #14: The M5 bar close bid/ask can be minutes old. For wide-spread
        # pairs like GBP_AUD, the live spread can differ significantly.
        live_bid, live_ask = self.client.get_price(pair)
        if live_bid and live_ask:
            bid, ask = live_bid, live_ask  # use live prices for spread check and sizing

        ok, spread = self.check_spread(pair, bid, ask)
        if not ok:
            logging.info(f'  {pair}: LIVE spread {spread:.1f} > max {Config.MAX_SPREAD_PIPS} — skipping')
            return

        # Check max concurrent
        if len(self.tracker.open_trades) >= Config.MAX_CONCURRENT:
            logging.debug(f'  {pair}: max concurrent reached ({Config.MAX_CONCURRENT})')
            return

        # Calculate position size
        equity = self.get_equity()
        if equity <= 0:
            logging.error('Cannot size: equity is 0')
            return

        units = calc_units(pair, direction, equity, bid, ask)
        if abs(units) < Config.MIN_UNITS:
            logging.warning(f'  {pair}: units too small ({units})')
            return

        # Place order
        comment = f'ACID_{pair}_{direction}'
        fill = self.client.place_market(pair, units, comment)
        if fill:
            fill_price = float(fill.get('price', ask if direction > 0 else bid))
            trade_id = None
            if 'tradeOpened' in fill:
                trade_id = fill['tradeOpened'].get('tradeID')
            if not trade_id:
                trade_id = fill.get('id', 'unknown')

            # FIX #2: Store real UTC entry time
            entry_time = datetime.now(timezone.utc)
            self.tracker.add_trade(pair, trade_id, direction, fill_price, entry_time)
            self.stats['entries'] += 1
            dir_label = "LONG" if direction > 0 else "SHORT"
            if Config.SPREAD_BET:
                size_label = f'£{abs(units):.2f}/pip'
                mid = (bid + ask) / 2.0
                pip_sz = 0.01 if 'JPY' in pair else 0.0001
                est_margin = abs(units) * (mid / pip_sz) / 30.0
                margin_label = f'est_margin=£{est_margin:.2f}'
            else:
                size_label = f'{abs(units)} units'
                margin_label = f'margin=£{equity*Config.EQUITY_PER_TRADE_PCT:.2f}'
            logging.info(f'  ENTRY {pair} {dir_label} | '
                         f'{size_label} @ {fill_price:.5f} | '
                         f'spread {spread:.1f} | equity £{equity:.2f}')
            self.journal.info(
                f'  {pair}: FILLED {dir_label} | {size_label} fill={fill_price:.5f} '
                f'spread={spread:.1f} equity=£{equity:.2f} {margin_label} '
                f'trade_id={trade_id}')
        else:
            self.stats['errors'] += 1
            logging.error(f'  ENTRY FAILED {pair}')
            self.journal.error(f'  {pair}: ORDER REJECTED by OANDA')

    def try_exit(self, pair, info, reason):
        """Attempt to close a trade."""
        trade_id = info['trade_id']
        equity_before = self.get_equity()

        result = self.client.close_trade(trade_id)
        if result:
            exit_price = float(result.get('price', 0))
            equity_after = self.get_equity()

            self.trade_log.log_trade(
                pair, info['direction'], 0,
                info['entry_price'], exit_price,
                info['entry_time_utc'], reason,
                equity_before, equity_after,
            )
            self.tracker.remove_trade(pair)
            self.stats['exits'] += 1

            pm = pip_mult(pair)
            pnl_pips = info['direction'] * (exit_price - info['entry_price']) * pm
            self.journal.info(
                f'  {pair}: CLOSED | reason={reason} | pnl={pnl_pips:+.1f} pips | '
                f'entry={info["entry_price"]:.5f} exit={exit_price:.5f} | '
                f'equity £{equity_before:.2f} -> £{equity_after:.2f} ({equity_after-equity_before:+.2f})')
        else:
            self.stats['errors'] += 1
            logging.error(f'  EXIT FAILED {pair} (trade {trade_id})')
            self.journal.error(f'  {pair}: CLOSE FAILED trade_id={trade_id}')

    def sync_with_broker(self):
        """
        Sync tracked positions with what's actually open at OANDA.
        Handles manual closes, margin calls, etc.
        """
        broker_trades = self.client.get_open_trades()
        broker_pairs = {}
        for t in broker_trades:
            inst = t.get('instrument', '')
            broker_pairs[inst] = t

        tracked_pairs = set(self.tracker.open_trades.keys())

        # Trades we think are open but broker doesn't have
        for pair in tracked_pairs - set(broker_pairs.keys()):
            logging.warning(f'  SYNC: {pair} closed externally, removing from tracker')
            self.tracker.remove_trade(pair)

    def is_market_open(self):
        """Check if forex market is open (skip Sat/Sun)."""
        now = datetime.now(timezone.utc)
        if now.weekday() == 5:  # Saturday
            return False
        if now.weekday() == 6 and now.hour < 22:  # Sunday before open
            return False
        if now.weekday() == 4 and now.hour >= 22:  # Friday after close
            return False
        return True

    def run_cycle(self):
        """One full cycle: check all pairs, manage exits and entries."""
        pair_states = {}
        pair_biases = {}
        pair_spreads = {}
        new_transitions = []

        # Cycle-level rejection counters for summary
        cycle_stats = {
            'scanned': 0, 'new_bars': 0, 'transitions': 0,
            'skip_no_bar': 0, 'skip_same_bar': 0,
            'reject_toxic': 0, 'reject_spread': 0, 'reject_positioned': 0,
            'reject_max_conc': 0, 'reject_just_exited': 0,
            'entries': 0, 'exits': 0,
        }

        now_hour = datetime.now(timezone.utc).hour
        is_toxic = now_hour in Config.TOXIC_HOURS_UTC

        self.journal.info('─' * 80)
        self.journal.info(f'CYCLE START | {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")} UTC | '
                          f'hour={now_hour} {"TOXIC" if is_toxic else "OK"} | '
                          f'open_positions={len(self.tracker.open_trades)}')

        # Phase 1: Compute regime state for all pairs
        for pair in Config.PAIRS:
            cycle_stats['scanned'] += 1
            try:
                state, bid, ask, bar_time, bias_val, spread = self.get_regime_state(pair)
                pair_states[pair] = state
                pair_biases[pair] = bias_val
                pair_spreads[pair] = spread

                if bar_time is None:
                    cycle_stats['skip_no_bar'] += 1
                    self.journal.debug(f'  {pair}: no bar data')
                    continue

                # Detect new bar (only act on new completed bars)
                is_new_bar = False
                prev_time = self.last_m5_time.get(pair)
                if prev_time is None or bar_time > prev_time:
                    is_new_bar = True
                    self.last_m5_time[pair] = bar_time
                    cycle_stats['new_bars'] += 1

                    # FIX #10: Increment bar counter for open trades on this pair.
                    # This counts actual market-time M5 bars (weekends skipped
                    # automatically since no bars are generated).
                    if self.tracker.has_position(pair):
                        self.tracker.increment_bars_held(pair)
                else:
                    cycle_stats['skip_same_bar'] += 1

                # Journal every pair's state
                prev = self.prev_state.get(pair, 0)
                state_name = {1: 'BULL', -1: 'BEAR', 0: 'FLAT'}.get(state, '?')
                prev_name = {1: 'BULL', -1: 'BEAR', 0: 'FLAT'}.get(prev, '?')
                has_pos = self.tracker.has_position(pair)

                self.journal.debug(
                    f'  {pair}: bias={bias_val:+.4f} state={state_name} '
                    f'prev={prev_name} spread={spread:.1f} '
                    f'new_bar={is_new_bar} positioned={has_pos} '
                    f'bar={bar_time.strftime("%H:%M") if bar_time else "?"}')

                # Only detect transitions on NEW completed bars
                # FIX #8: Do NOT update prev_state here — defer to after Phase 3.
                # If a transition is rejected for retry-able reasons (toxic hours,
                # spread, max concurrent), we must NOT update prev_state so the
                # transition can be re-detected on the next cycle when conditions
                # improve. Updating here permanently loses the transition.
                if is_new_bar:
                    if prev == 0 and state != 0:
                        cycle_stats['transitions'] += 1
                        dir_label = "LONG" if state == 1 else "SHORT"
                        self.journal.info(
                            f'  {pair}: *** TRANSITION 0->{state_name} *** | '
                            f'bias={bias_val:+.4f} spread={spread:.1f} '
                            f'bid={bid:.5f} ask={ask:.5f}')
                        new_transitions.append((pair, state, bid, ask))
                    else:
                        # No transition detected — safe to update prev_state
                        self.prev_state[pair] = state

            except Exception as e:
                logging.error(f'  {pair} signal error: {e}')
                self.journal.error(f'  {pair}: EXCEPTION: {e}')
                self.stats['errors'] += 1

        # Phase 2: Process exits first (free up pairs before entries)
        to_close = self.tracker.get_trades_to_close(pair_states)
        exited_this_cycle = set()
        for pair, info, reason in to_close:
            try:
                bias_val = pair_biases.get(pair)
                bias_str = f'{bias_val:+.4f}' if bias_val is not None else '?'
                self.journal.info(
                    f'  {pair}: EXIT | reason={reason} | bias={bias_str} | '
                    f'entry_price={info["entry_price"]:.5f}')
                self.try_exit(pair, info, reason)
                exited_this_cycle.add(pair)
                cycle_stats['exits'] += 1
            except Exception as e:
                logging.error(f'  {pair} exit error: {e}')
                self.journal.error(f'  {pair}: EXIT EXCEPTION: {e}')
                self.stats['errors'] += 1

        # Phase 3: Process entries (only regime transitions)
        # FIX #8: Track which transitions were permanently consumed vs retry-able.
        # Retry-able rejections (toxic hours, spread, max concurrent) should NOT
        # update prev_state, so the transition can be re-detected next cycle.
        # Permanent rejections (already positioned, just exited) and successful
        # entries SHOULD update prev_state.
        transition_pairs_consumed = set()   # pairs where entry executed or permanently skipped
        transition_pairs_retryable = set()  # pairs where we should retry next cycle

        for pair, direction, bid, ask in new_transitions:
            try:
                dir_label = "LONG" if direction == 1 else "SHORT"
                spread = pair_spreads.get(pair, 0)
                bias_val = pair_biases.get(pair)

                # Log the full decision chain
                if self.tracker.has_position(pair):
                    self.journal.info(f'  {pair}: SKIP {dir_label} | reason=ALREADY_POSITIONED')
                    cycle_stats['reject_positioned'] += 1
                    transition_pairs_consumed.add(pair)
                    continue

                if pair in exited_this_cycle:
                    self.journal.info(f'  {pair}: SKIP {dir_label} | reason=JUST_EXITED_THIS_CYCLE')
                    cycle_stats['reject_just_exited'] += 1
                    transition_pairs_consumed.add(pair)
                    continue

                bias_str = f'{bias_val:+.4f}' if bias_val is not None else '?'

                if is_toxic:
                    self.journal.info(
                        f'  {pair}: SKIP {dir_label} | reason=TOXIC_HOUR ({now_hour}:00 UTC) | '
                        f'bias={bias_str} spread={spread:.1f} | WILL RETRY when toxic hours end')
                    cycle_stats['reject_toxic'] += 1
                    transition_pairs_retryable.add(pair)
                    continue

                if spread > Config.MAX_SPREAD_PIPS:
                    self.journal.info(
                        f'  {pair}: SKIP {dir_label} | reason=SPREAD ({spread:.1f} > {Config.MAX_SPREAD_PIPS}) | '
                        f'bias={bias_str} | WILL RETRY when spread narrows')
                    cycle_stats['reject_spread'] += 1
                    transition_pairs_retryable.add(pair)
                    continue

                if len(self.tracker.open_trades) >= Config.MAX_CONCURRENT:
                    self.journal.info(f'  {pair}: SKIP {dir_label} | reason=MAX_CONCURRENT | WILL RETRY when slot frees')
                    cycle_stats['reject_max_conc'] += 1
                    transition_pairs_retryable.add(pair)
                    continue

                if bid and ask:
                    self.journal.info(
                        f'  {pair}: ENTER {dir_label} | '
                        f'bias={bias_str} spread={spread:.1f} '
                        f'bid={bid:.5f} ask={ask:.5f}')
                    self.try_entry(pair, direction, bid, ask)
                    cycle_stats['entries'] += 1
                    transition_pairs_consumed.add(pair)
            except Exception as e:
                logging.error(f'  {pair} entry error: {e}')
                self.journal.error(f'  {pair}: ENTRY EXCEPTION: {e}')
                self.stats['errors'] += 1
                transition_pairs_consumed.add(pair)

        # Phase 4: Update prev_state for transition pairs
        # FIX #8: Only update prev_state for transitions that were consumed
        # (entered or permanently rejected). Retry-able rejections keep
        # prev_state=0 so the transition is re-detected next cycle.
        for pair, direction, bid, ask in new_transitions:
            if pair in transition_pairs_consumed:
                self.prev_state[pair] = pair_states.get(pair, 0)
            elif pair in transition_pairs_retryable:
                # Don't update — keep prev_state as-is so transition re-fires
                self.journal.debug(
                    f'  {pair}: prev_state NOT updated (retry-able rejection)')
            else:
                # Fallback: update normally
                self.prev_state[pair] = pair_states.get(pair, 0)

        # Cycle summary — one line to main log, detail to journal
        n_trans = cycle_stats['transitions']
        n_entries = cycle_stats['entries']
        n_exits = cycle_stats['exits']
        rejects = (cycle_stats['reject_toxic'] + cycle_stats['reject_spread'] +
                   cycle_stats['reject_positioned'] + cycle_stats['reject_just_exited'] +
                   cycle_stats['reject_max_conc'])

        # Build reject breakdown string when there are rejects
        if rejects:
            parts = []
            if cycle_stats['reject_toxic']:
                parts.append(f'toxic={cycle_stats["reject_toxic"]}')
            if cycle_stats['reject_spread']:
                parts.append(f'spread={cycle_stats["reject_spread"]}')
            if cycle_stats['reject_positioned']:
                parts.append(f'positioned={cycle_stats["reject_positioned"]}')
            if cycle_stats['reject_just_exited']:
                parts.append(f'just_exited={cycle_stats["reject_just_exited"]}')
            if cycle_stats['reject_max_conc']:
                parts.append(f'max_conc={cycle_stats["reject_max_conc"]}')
            reject_detail = f'rejects={rejects}({",".join(parts)})'
        else:
            reject_detail = 'rejects=0'

        logging.info(
            f'  scan={cycle_stats["scanned"]} new_bars={cycle_stats["new_bars"]} '
            f'transitions={n_trans} entries={n_entries} exits={n_exits} '
            f'{reject_detail} open={len(self.tracker.open_trades)}')

        self.journal.info(
            f'CYCLE END | scanned={cycle_stats["scanned"]} new_bars={cycle_stats["new_bars"]} '
            f'transitions={n_trans} entries={n_entries} exits={n_exits} | '
            f'reject_toxic={cycle_stats["reject_toxic"]} reject_spread={cycle_stats["reject_spread"]} '
            f'reject_positioned={cycle_stats["reject_positioned"]} '
            f'reject_exited={cycle_stats["reject_just_exited"]} '
            f'reject_max_conc={cycle_stats["reject_max_conc"]}')
        self.journal.info('')

        # FIX #10: Persist bar counters once per cycle (not per-bar, avoids disk thrashing)
        if cycle_stats['new_bars'] > 0:
            self.tracker.save_state_if_changed()

    def print_status(self):
        equity = self.get_equity()
        n_open = len(self.tracker.open_trades)
        logging.info(f'── Status: £{equity:.2f} | {n_open} positions | '
                     f'entries={self.stats["entries"]} exits={self.stats["exits"]} '
                     f'errors={self.stats["errors"]} ──')
        if n_open > 0:
            now = datetime.now(timezone.utc)
            for pair, info in self.tracker.open_trades.items():
                entry_time = self.tracker.get_entry_time(pair)
                if entry_time and entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                held = (now - entry_time).total_seconds() / 3600.0 if entry_time else 0
                bars = info.get('bars_held', 0)
                logging.info(f'    {pair} {"L" if info["direction"]>0 else "S"} '
                             f'@ {info["entry_price"]:.5f} | {bars}/{Config.MAX_HOLD_BARS} bars | {held:.1f}h wall')

    def run(self):
        """Main loop — polls every M5 bar."""
        self.setup_logging()

        logging.info('=' * 70)
        logging.info('  REGIME ACID TEST BOT v3 — LIVE (forensic-fixed)')
        logging.info('=' * 70)
        logging.info(f'  EMA engine: {"talib (SMA-seeded)" if HAS_TALIB else "numpy (first-value seeded)"}')
        logging.info(f'  Incomplete higher-TF bar protection: ENABLED (FIX #7)')
        logging.info(f'  Retry-able transition preservation: ENABLED (FIX #8)')
        logging.info(f'  Real M1 data (not faked from M5): ENABLED (FIX #11)')
        logging.info(f'  Persistent candle cache: ENABLED (FIX #12)')
        logging.info(f'  Max hold: {Config.MAX_HOLD_BARS} M5 bars (market time, weekends skipped) (FIX #10)')

        # Validate connection
        acct = self.client.get_account()
        if not acct:
            logging.error('FATAL: Cannot connect to OANDA. Check credentials.')
            sys.exit(1)

        equity = float(acct.get('NAV', acct.get('balance', 0)))
        logging.info(f'  Account: {Config.OANDA_ACCOUNT_ID}')
        logging.info(f'  Equity:  £{equity:.2f}')
        logging.info(f'  Environment: {Config.OANDA_ENVIRONMENT}')

        # ── Auto-detect account type from instrument metadata ──
        inst_info = self.client.get_instrument_info('EUR_USD')
        if inst_info:
            tu_prec = inst_info.get('tradeUnitsPrecision', 0)
            min_sz = inst_info.get('minimumTradeSize', '1')
            logging.info(f'  EUR_USD tradeUnitsPrecision={tu_prec} minimumTradeSize={min_sz}')

            if Config.SPREAD_BET is None:
                # Auto-detect: precision > 0 means spread bet, 0 means CFD
                if tu_prec > 0:
                    Config.SPREAD_BET = True
                    Config.MIN_UNITS = float(min_sz) if min_sz != '?' else 0.01
                    logging.info(f'  Auto-detected: SPREAD BET (tradeUnitsPrecision={tu_prec})')
                else:
                    Config.SPREAD_BET = False
                    Config.MIN_UNITS = int(float(min_sz)) if min_sz != '?' else 1
                    logging.info(f'  Auto-detected: CFD (tradeUnitsPrecision={tu_prec})')
            else:
                # Manual override — warn if mismatch
                if Config.SPREAD_BET and tu_prec == 0:
                    logging.warning('  WARNING: SPREAD_BET=True but tradeUnitsPrecision=0 — '
                                    'this looks like a CFD account!')
                elif not Config.SPREAD_BET and tu_prec > 0:
                    logging.warning('  WARNING: SPREAD_BET=False but tradeUnitsPrecision>0 — '
                                    'this looks like a spread bet account!')
        else:
            # Couldn't fetch instrument info — fall back to CFD if not set
            if Config.SPREAD_BET is None:
                Config.SPREAD_BET = False
                Config.MIN_UNITS = 1
                logging.warning('  Could not fetch instrument info — defaulting to CFD mode')

        acct_type = 'SPREAD BET (GBP/pip)' if Config.SPREAD_BET else 'CFD (base currency units)'
        logging.info(f'  Account type: {acct_type}')
        logging.info(f'  ET={Config.ENTRY_THRESHOLD} XT={Config.EXIT_THRESHOLD} '
                     f'TE={Config.MAX_HOLD_BARS} bars (={Config.MAX_HOLD_BARS*5/60:.0f}h market time)')
        logging.info(f'  Sizing: {Config.EQUITY_PER_TRADE_PCT*100:.1f}% equity/trade, '
                     f'min={Config.MIN_UNITS} {"£/pip" if Config.SPREAD_BET else "units"}')
        logging.info(f'  Pairs: {len(Config.PAIRS)}')
        logging.info(f'  Max spread: {Config.MAX_SPREAD_PIPS} pips')
        logging.info(f'  Toxic hours: {Config.TOXIC_HOURS_UTC} UTC (no entries)')
        logging.info(f'  M5 history: {Config.CANDLE_HISTORY} bars (~{Config.CANDLE_HISTORY*5/60/24:.0f} days)')
        logging.info(f'  M1 history: {Config.M1_CANDLE_HISTORY} bars (~{Config.M1_CANDLE_HISTORY/60/24:.0f} days) (FIX #13: paginated to match M5)')
        logging.info(f'  Cache dir: {Config.CACHE_DIR}')

        if Config.SPREAD_BET:
            sample_stake = equity * Config.EQUITY_PER_TRADE_PCT * 30.0 * 0.0001 / 1.08
            sample_stake = max(round(sample_stake, 2), Config.MIN_UNITS)
            logging.info(f'  Example: EUR/USD @ 1.08 → £{sample_stake:.2f}/pip, '
                         f'margin ≈ £{sample_stake * 10800 / 30:.2f}')
        else:
            sample_units = int(equity * Config.EQUITY_PER_TRADE_PCT * 30.0)
            logging.info(f'  Example: {sample_units} units/trade, '
                         f'pip value ≈ £{sample_units * 0.0001 / 1.26:.4f} (EUR/USD)')
        logging.info(f'  19 pairs × ~£{equity * Config.EQUITY_PER_TRADE_PCT:.2f} margin = '
                     f'~£{19 * equity * Config.EQUITY_PER_TRADE_PCT:.0f} total '
                     f'({19 * Config.EQUITY_PER_TRADE_PCT * 100:.0f}% of equity)')
        logging.info('')

        # FIX #2: Try to recover state from disk first (survives restarts)
        state_loaded = self.tracker._load_state()

        if state_loaded and self.tracker.open_trades:
            # Verify recovered trades still exist at OANDA
            logging.info('  Verifying recovered trades against OANDA...')
            broker_trades = self.client.get_open_trades()
            broker_ids = {t.get('id') for t in broker_trades}
            stale = []
            for pair, info in self.tracker.open_trades.items():
                if info['trade_id'] not in broker_ids:
                    stale.append(pair)
                    logging.warning(f'    {pair}: trade {info["trade_id"]} no longer open at OANDA')
                else:
                    entry_time = self.tracker.get_entry_time(pair)
                    if entry_time and entry_time.tzinfo is None:
                        entry_time = entry_time.replace(tzinfo=timezone.utc)
                    held = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600.0 if entry_time else 0
                    bars = info.get('bars_held', 0)
                    # Ensure bars_held field exists (backwards compat with old state files)
                    if 'bars_held' not in info:
                        info['bars_held'] = 0
                    logging.info(f'    Recovered: {pair} {"L" if info["direction"]>0 else "S"} '
                                 f'@ {info["entry_price"]:.5f} | {bars} bars / {held:.1f}h wall | trade {info["trade_id"]}')
            for pair in stale:
                self.tracker.remove_trade(pair)
        else:
            # Check for existing OANDA positions from a previous run
            logging.info('  Checking for existing positions at OANDA...')
            existing = self.client.get_open_trades()
            recovered = 0
            for t in existing:
                pair = t.get('instrument', '')
                if pair in Config.PAIRS:
                    units = float(t.get('currentUnits', t.get('initialUnits', 0)))
                    direction = 1 if units > 0 else -1
                    trade_id = t.get('id')
                    entry_price = float(t.get('price', 0))
                    # Use OANDA's trade open time
                    open_time_str = t.get('openTime', '')
                    if open_time_str:
                        entry_time = pd.Timestamp(open_time_str).to_pydatetime()
                    else:
                        entry_time = datetime.now(timezone.utc)
                    self.tracker.add_trade(pair, trade_id, direction, entry_price, entry_time)
                    recovered += 1
                    logging.info(f'    Recovered: {pair} {"L" if direction>0 else "S"} '
                                 f'@ {entry_price:.5f} (trade {trade_id})')
            if recovered:
                logging.info(f'  Recovered {recovered} existing positions')
            else:
                logging.info('  No existing positions')

        # FIX #12: Warm up candle cache BEFORE any trading decisions.
        # Load cached data from disk first (instant), then fetch only missing bars.
        logging.info('  Loading candle cache from disk...')
        cached_files = self.cache.load_all(Config.PAIRS)
        if cached_files > 0:
            logging.info(f'  Loaded {cached_files} cached files from {Config.CACHE_DIR}/')
            logging.info(self.cache.summary())
        else:
            logging.info('  No cache found — will fetch full history for all pairs.')

        # Fetch missing/new data for each pair (M5 and M1)
        logging.info('  Warming up candle data (M5 + M1 for all pairs)...')
        for pair in Config.PAIRS:
            try:
                # This fetches incremental data if cache exists, or full history if not
                self.update_cache_for_pair(pair)
                m5_bars = self.cache.bar_count(pair, 'M5')
                m1_bars = self.cache.bar_count(pair, 'M1')
                logging.info(f'    {pair}: M5={m5_bars} bars | M1={m1_bars} bars')
            except Exception as e:
                logging.error(f'    {pair}: cache warmup error: {e}')
            time.sleep(0.2)  # don't hammer API

        logging.info('  Cache warmup complete.')
        logging.info(f'  Cache summary:\n{self.cache.summary()}')

        # Initialise prev_state for all pairs (avoid false entries on first cycle)
        logging.info('  Initialising regime states from cached data...')
        for pair in Config.PAIRS:
            try:
                state, bid, ask, bar_time, bias_val, spread = self.get_regime_state(pair)
                self.prev_state[pair] = state
                if bar_time is not None:
                    self.last_m5_time[pair] = bar_time
                state_name = {1: 'BULL', -1: 'BEAR', 0: 'FLAT'}.get(state, '?')
                bias_str = f'bias={bias_val:+.4f}' if bias_val is not None else 'bias=?'
                spread_str = f'spread={spread:.1f}' if spread is not None else 'spread=?'
                logging.info(f'    {pair}: {state_name} | {bias_str} | {spread_str}')
            except Exception as e:
                logging.error(f'    {pair}: init error: {e}')
                self.prev_state[pair] = 0

        logging.info(f'\n  Initialised. Waiting for M5 bar boundaries...\n')

        # Set up graceful shutdown
        def shutdown(signum, frame):
            logging.info('\n  SHUTDOWN requested — closing all positions...')
            self.running = False
            for pair, info in list(self.tracker.open_trades.items()):
                self.try_exit(pair, info, 'SHUTDOWN')
            logging.info('  All positions closed. Exiting.')
            sys.exit(0)

        sig_module.signal(sig_module.SIGINT, shutdown)
        sig_module.signal(sig_module.SIGTERM, shutdown)

        # Main loop — wait for M5 bar boundaries
        last_m5_minute = -1
        cycle_count = 0
        consecutive_errors = 0
        last_heartbeat = datetime.now(timezone.utc)

        while self.running:
            try:
                now = datetime.now(timezone.utc)

                # Heartbeat every 30 min so logs prove the bot is alive
                if (now - last_heartbeat).total_seconds() >= 1800:
                    last_heartbeat = now
                    n_open = len(self.tracker.open_trades)
                    logging.info(f'  HEARTBEAT | {now.strftime("%Y-%m-%d %H:%M")} UTC | '
                                 f'{n_open} positions | errors_streak={consecutive_errors}')

                # Skip weekends
                if not self.is_market_open():
                    if cycle_count > 0:
                        logging.info('  Market closed (weekend). Sleeping 5min...')
                        cycle_count = 0
                    time.sleep(300)
                    continue

                m5_minute = (now.minute // 5) * 5

                if m5_minute != last_m5_minute:
                    last_m5_minute = m5_minute

                    # Wait 15s after bar close for OANDA to finalise candles
                    bar_close = now.replace(minute=m5_minute, second=0, microsecond=0)
                    wait_until = bar_close + timedelta(seconds=15)
                    sleep_secs = max(0, (wait_until - datetime.now(timezone.utc)).total_seconds())
                    if sleep_secs > 0:
                        time.sleep(sleep_secs)

                    logging.info(f'\n── M5 bar {now.strftime("%H:%M")} UTC ──')

                    # Sync with broker
                    self.sync_with_broker()

                    # Run the cycle
                    self.run_cycle()

                    # Status every 6 bars (30 min)
                    cycle_count += 1
                    if cycle_count % 6 == 0:
                        self.print_status()

                    consecutive_errors = 0  # reset on successful cycle

                time.sleep(Config.CHECK_INTERVAL_SECS)

            except Exception as e:
                consecutive_errors += 1
                logging.error(f'  MAIN LOOP ERROR (#{consecutive_errors}): {e}')
                logging.error(f'  Traceback: {__import__("traceback").format_exc()}')
                # Back off on repeated errors, but NEVER die
                backoff = min(60, 10 * consecutive_errors)
                logging.error(f'  Retrying in {backoff}s...')
                time.sleep(backoff)


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    bot = AcidTestBot()
    bot.run()
