#!/usr/bin/env python3
"""
BinanceDataProvider — Binance USDT-M TradFi永续合约数据接口
为 StockAnalysis 和 CANN 提供美股/ETF/商品永续的统一数据层。

支持的接口：
- K线（OHLCV）：get_klines()
- 实时价格：get_price()
- 资金费率：get_funding_rate(), get_funding_rate_history()
- 持仓量OI：get_open_interest()
- 技术指标：get_technical_indicators()

前置依赖：pip install python-binance
"""

import os
import sys
import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('BinanceData')

_client_class = None

def _get_client_class():
    global _client_class
    if _client_class is None:
        try:
            from binance.client import Client
            _client_class = Client
        except ImportError:
            try:
                from binance.um_futures import UMFutures
                _client_class = UMFutures
            except ImportError:
                raise ImportError("pip install python-binance")
    return _client_class


# ============================================================
# TradFi 品种列表（Binance USDT-M 永续）
# ============================================================

# 美股个股永续（~32只已知）
SYMBOLS = [
    # Mag 7 科技巨头 (0-6)
    "NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "AMZNUSDT", "GOOGLUSDT",
    "METAUSDT", "TSLAUSDT",
    # 半导体 (7-10)
    "INTCUSDT", "AMDUSDT", "AVGOUSDT", "QCOMUSDT",
    # 加密相关 (11-15)
    "MSTRUSDT", "COINUSDT", "CRCLUSDT", "HOODUSDT", "PLTRUSDT",
    # 企业科技 (16-19)
    "ORCLUSDT", "CSCOUSDT", "UBERUSDT", "SOFIUSDT",
    # 消费/零售 (20-22)
    "DISUSDT", "HDUSDT", "SBUXUSDT",
    # 医药 (23-24)
    "LLYUSDT", "NVSUSDT",
    # ETF (25-29)
    "SPYUSDT", "QQQUSDT", "SOXLUSDT", "GLDUSDT", "IBITUSDT",
    # 商品 (30-32)
    "XAUUSDT", "XAGUSDT", "CLUSDT",
    # Pre-IPO (33-34)
    "SPACEXUSDT", "OPENAIUSDT",
]

VARIETY_NAMES = {i: s.replace("USDT", "") for i, s in enumerate(SYMBOLS)}
VARIETY_CODES = {i: s for i, s in enumerate(SYMBOLS)}
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
NUM_VARIETIES = len(SYMBOLS)


def get_variety_info(vid: int) -> Tuple[str, str, str]:
    """返回 (名称, 代码, 类别)"""
    categories = {
        **{i: "科技巨头" for i in range(7)},
        **{i: "半导体" for i in range(7, 11)},
        **{i: "加密相关" for i in range(11, 16)},
        **{i: "企业科技" for i in range(16, 20)},
        **{i: "消费零售" for i in range(20, 23)},
        **{i: "医药" for i in range(23, 25)},
        **{i: "ETF" for i in range(25, 30)},
        **{i: "商品" for i in range(30, 33)},
        **{i: "Pre-IPO" for i in range(33, 35)},
    }
    return VARIETY_NAMES.get(vid, ""), VARIETY_CODES.get(vid, ""), categories.get(vid, "Other")


# ============================================================
# BinanceDataProvider
# ============================================================

class BinanceDataProvider:
    """Binance USDT-M 永续统一数据接口"""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET', '')
        self._client = None
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}

    def _get_cache(self, key: str, ttl: float = 300) -> Optional[Any]:
        if key in self._cache and key in self._cache_ttl:
            if time.time() - self._cache_ttl[key] < ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ttl[key] = time.time()

    def _get_client(self):
        if self._client is None:
            Client = _get_client_class()
            self._client = Client(self.api_key, self.api_secret) if self.api_key else Client()
        return self._client

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def close(self):
        if self._client:
            try: self._client.close_connection()
            except Exception: pass
            self._client = None

    # ---- K线 ----
    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 200) -> List[Dict]:
        cache_key = f"klines_{symbol}_{interval}_{limit}"
        cached = self._get_cache(cache_key, ttl=60)
        if cached: return cached

        client = self._get_client()
        try:
            raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            klines = [{
                'open_time': k[0], 'open': float(k[1]), 'high': float(k[2]),
                'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5]),
                'close_time': k[6], 'quote_volume': float(k[7]), 'trades': k[8],
            } for k in raw]
            self._set_cache(cache_key, klines)
            return klines
        except Exception as e:
            logger.error(f"K线获取失败 {symbol}: {e}")
            return []

    # ---- 价格 ----
    def get_price(self, symbol: str) -> Optional[float]:
        cache_key = f"price_{symbol}"
        cached = self._get_cache(cache_key, ttl=10)
        if cached: return cached
        try:
            ticker = self._get_client().futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            self._set_cache(cache_key, price)
            return price
        except Exception as e:
            logger.error(f"价格获取失败 {symbol}: {e}")
            return None

    # ---- 资金费率 ----
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        cache_key = f"funding_{symbol}"
        cached = self._get_cache(cache_key, ttl=120)
        if cached: return cached
        try:
            raw = self._get_client().futures_funding_rate(symbol=symbol, limit=1)
            if raw:
                data = {'symbol': raw[0]['symbol'],
                        'funding_rate': float(raw[0]['fundingRate']),
                        'funding_time': raw[0].get('fundingTime', 0),
                        'mark_price': float(raw[0].get('markPrice', 0))}
                self._set_cache(cache_key, data)
                return data
        except Exception as e:
            logger.error(f"资金费率获取失败 {symbol}: {e}")
        return None

    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        cache_key = f"funding_hist_{symbol}_{limit}"
        cached = self._get_cache(cache_key, ttl=600)
        if cached: return cached
        try:
            raw = self._get_client().futures_funding_rate(symbol=symbol, limit=limit)
            result = [{'funding_rate': float(r['fundingRate']),
                       'funding_time': r.get('fundingTime', 0)} for r in raw]
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"历史费率获取失败 {symbol}: {e}")
            return []

    def get_funding_rate_percentile(self, symbol: str, window: int = 100) -> Tuple[float, float]:
        current = self.get_funding_rate(symbol)
        if not current: return 0.0, 50.0
        history = self.get_funding_rate_history(symbol, limit=window)
        if len(history) < 10: return current['funding_rate'], 50.0
        rates = sorted([h['funding_rate'] for h in history])
        n_lower = sum(1 for r in rates if r <= current['funding_rate'])
        return current['funding_rate'], (n_lower / len(rates)) * 100

    def get_all_funding_rates(self) -> Dict[str, Dict]:
        try:
            raw = self._get_client().futures_funding_rate(limit=500)
            result = {}
            for r in raw:
                sym = r['symbol']
                if sym in SYMBOLS:
                    result[sym] = {'funding_rate': float(r['fundingRate']),
                                   'funding_time': r.get('fundingTime', 0)}
            return result
        except Exception as e:
            logger.error(f"批量费率获取失败: {e}")
            return {}

    def get_all_funding_rates_dict(self) -> Dict[int, float]:
        all_rates = self.get_all_funding_rates()
        return {SYMBOL_TO_ID[sym]: d['funding_rate']
                for sym, d in all_rates.items() if sym in SYMBOL_TO_ID}

    # ---- OI ----
    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        cache_key = f"oi_{symbol}"
        cached = self._get_cache(cache_key, ttl=120)
        if cached: return cached
        try:
            raw = self._get_client().futures_open_interest(symbol=symbol)
            data = {'symbol': raw['symbol'],
                    'open_interest': float(raw['openInterest']),
                    'time': raw.get('time', 0)}
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"OI获取失败 {symbol}: {e}")
            return None

    def get_all_open_interests(self) -> Dict[int, float]:
        try:
            raw = self._get_client().futures_open_interest(limit=500)
            return {SYMBOL_TO_ID[r['symbol']]: float(r['openInterest'])
                    for r in raw if r['symbol'] in SYMBOL_TO_ID}
        except Exception as e:
            logger.error(f"批量OI获取失败: {e}")
            return {}

    # ---- 技术指标 ----
    def get_technical_indicators(self, symbol: str, interval: str = "1d",
                                  limit: int = 200) -> Dict:
        cache_key = f"tech_{symbol}_{interval}_{limit}"
        cached = self._get_cache(cache_key, ttl=120)
        if cached: return cached

        klines = self.get_klines(symbol, interval=interval, limit=limit)
        if len(klines) < 50:
            return {'price': 0, 'ma20': 0, 'ma50': 0, 'ma200': 0,
                    'bb_upper': 0, 'bb_lower': 0, 'atr': 0, 'atr_pct': 0,
                    'rsi14': 50, 'high_20d': 0, 'low_20d': 0,
                    'price_percentile_200d': 0.5, 'ma_alignment': 'mixed'}

        closes = np.array([k['close'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])
        current = closes[-1]

        def ma(d, p): return float(np.mean(d[-p:])) if len(d) >= p else float(d[-1])

        ma20, ma50 = ma(closes, 20), ma(closes, 50)
        ma200 = ma(closes, 200) if len(closes) >= 200 else ma50

        bb_mid = ma20
        bb_std = float(np.std(closes[-20:])) if len(closes) >= 20 else 0
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        trs = []
        for i in range(1, min(15, len(klines))):
            trs.append(max(highs[-i]-lows[-i], abs(highs[-i]-closes[-i-1]), abs(lows[-i]-closes[-i-1])))
        atr = float(np.mean(trs)) if trs else 0

        # RSI(14)
        rsi = 50.0
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain, avg_loss = np.mean(gains), np.mean(losses)
            if avg_loss > 0: rsi = float(100 - 100/(1 + avg_gain/avg_loss))

        high_20d = float(np.max(highs[-20:])) if len(highs) >= 20 else current
        low_20d = float(np.min(lows[-20:])) if len(lows) >= 20 else current

        if len(closes) >= 200:
            n_lower = sum(1 for c in closes[-200:] if c <= current)
            pct = float(n_lower/200)
        else:
            pct = 0.5

        if ma20 > ma50 > ma200: alignment = 'bullish'
        elif ma20 < ma50 < ma200: alignment = 'bearish'
        else: alignment = 'mixed'

        result = {
            'price': current, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_mid,
            'atr': atr, 'atr_pct': round(atr/current*100, 3) if current else 0,
            'rsi14': round(rsi, 1),
            'high_20d': high_20d, 'low_20d': low_20d,
            'price_percentile_200d': round(pct, 4), 'ma_alignment': alignment,
        }
        self._set_cache(cache_key, result)
        return result


# CLI自检
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    print("=" * 60)
    print("BinanceDataProvider — TradFi永续自检")
    print(f"品种总数: {NUM_VARIETIES}")
    print("=" * 60)

    with BinanceDataProvider() as p:
        for sym in ["NVDAUSDT", "SPYUSDT"]:
            price = p.get_price(sym)
            fr = p.get_funding_rate(sym)
            oi = p.get_open_interest(sym)
            tech = p.get_technical_indicators(sym)
            print(f"\n{sym}: ${price or 'N/A'}")
            print(f"  费率: {fr['funding_rate']*100:.4f}%" if fr else "  费率: N/A")
            print(f"  OI: {oi['open_interest']:,.0f}" if oi else "  OI: N/A")
            if tech['price']:
                print(f"  MA20=${tech['ma20']:.2f} MA50=${tech['ma50']:.2f} RSI={tech['rsi14']:.0f}")
                print(f"  布林: [${tech['bb_lower']:.2f}, ${tech['bb_upper']:.2f}] 排列={tech['ma_alignment']}")
