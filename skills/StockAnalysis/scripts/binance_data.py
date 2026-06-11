#!/usr/bin/env python3
"""
BinanceDataProvider — Binance USDT-M TradFi永续合约数据接口 (纯 requests 版)
为 StockAnalysis 和 SANN 提供美股/ETF/商品永续的统一数据层。

前置依赖：pip install requests numpy
"""

import os
import sys
import json
import time
import logging
import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger('BinanceData')

BASE_URL = "https://fapi.binance.com"


def _get(path: str, params: dict = None, timeout: int = 15) -> dict:
    """封装 Binance API GET 请求"""
    url = f"{BASE_URL}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and 'code' in data and data['code'] != 0:
            logger.error(f"API error: {data}")
            return {}
        return data
    except Exception as e:
        logger.error(f"请求失败 {path}: {e}")
        return {}


# ============================================================
# TradFi 品种列表
# ============================================================

SYMBOLS = [
    "NVDAUSDT", "AAPLUSDT", "MSFTUSDT", "AMZNUSDT", "GOOGLUSDT",
    "METAUSDT", "TSLAUSDT",
    "INTCUSDT", "AMDUSDT", "AVGOUSDT", "QCOMUSDT",
    "MSTRUSDT", "COINUSDT", "CRCLUSDT", "HOODUSDT", "PLTRUSDT",
    "ORCLUSDT", "CSCOUSDT", "UBERUSDT", "SOFIUSDT",
    "DISUSDT", "HDUSDT", "SBUXUSDT",
    "LLYUSDT", "NVSUSDT",
    "SPYUSDT", "QQQUSDT", "SOXLUSDT", "GLDUSDT", "IBITUSDT",
    "XAUUSDT", "XAGUSDT", "CLUSDT",
    "SPACEXUSDT", "OPENAIUSDT",
]

VARIETY_NAMES = {i: s.replace("USDT", "") for i, s in enumerate(SYMBOLS)}
VARIETY_CODES = {i: s for i, s in enumerate(SYMBOLS)}
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
NUM_VARIETIES = len(SYMBOLS)


def get_variety_info(vid: int) -> Tuple[str, str, str]:
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
class BinanceDataProvider:
    """Binance 数据接口 — 纯 requests 实现，零外部依赖"""

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}

    def _cached(self, key: str, ttl: float = 300) -> Optional[Any]:
        if key in self._cache and key in self._cache_ttl:
            if time.time() - self._cache_ttl[key] < ttl:
                return self._cache[key]
        return None

    def _set(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ttl[key] = time.time()

    def __enter__(self): return self
    def __exit__(self, *args): pass

    # ---- 价格 ----
    def get_price(self, symbol: str) -> Optional[float]:
        cached = self._cached(f"p_{symbol}", ttl=10)
        if cached: return cached
        data = _get("/fapi/v1/ticker/price", {"symbol": symbol})
        if data and 'price' in data:
            p = float(data['price'])
            self._set(f"p_{symbol}", p)
            return p
        return None

    def get_all_prices(self) -> Dict[str, float]:
        data = _get("/fapi/v1/ticker/price")
        if isinstance(data, list):
            return {p['symbol']: float(p['price']) for p in data if p['symbol'] in SYMBOLS}
        return {}

    # ---- K线 ----
    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 200) -> List[Dict]:
        key = f"kl_{symbol}_{interval}_{limit}"
        cached = self._cached(key, ttl=60)
        if cached: return cached

        data = _get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
        if not isinstance(data, list):
            return []

        klines = [{
            'open_time': k[0], 'open': float(k[1]), 'high': float(k[2]),
            'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5]),
            'close_time': k[6], 'quote_volume': float(k[7]), 'trades': k[8],
        } for k in data]
        self._set(key, klines)
        return klines

    # ---- 资金费率 ----
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        cached = self._cached(f"fr_{symbol}", ttl=120)
        if cached: return cached

        data = _get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
        if isinstance(data, list) and data:
            r = data[0]
            result = {'symbol': r['symbol'],
                      'funding_rate': float(r['fundingRate']),
                      'funding_time': r.get('fundingTime', 0),
                      'mark_price': float(r.get('markPrice', 0))}
            self._set(f"fr_{symbol}", result)
            return result
        return None

    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        key = f"frh_{symbol}_{limit}"
        cached = self._cached(key, ttl=600)
        if cached: return cached

        data = _get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
        if isinstance(data, list):
            result = [{'funding_rate': float(r['fundingRate']),
                       'funding_time': r.get('fundingTime', 0)} for r in data]
            self._set(key, result)
            return result
        return []

    def get_all_funding_rates(self) -> Dict[int, float]:
        """批量获取所有品种资金费率"""
        result = {}
        for sym in SYMBOLS:
            fr = self.get_funding_rate(sym)
            if fr:
                result[SYMBOL_TO_ID[sym]] = fr['funding_rate']
            time.sleep(0.05)  # 频率控制
        return result

    def get_all_funding_rates_dict(self) -> Dict[int, float]:
        return self.get_all_funding_rates()

    # ---- OI ----
    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        cached = self._cached(f"oi_{symbol}", ttl=120)
        if cached: return cached

        data = _get("/fapi/v1/openInterest", {"symbol": symbol})
        if data and 'openInterest' in data:
            result = {'symbol': data['symbol'],
                      'open_interest': float(data['openInterest']),
                      'time': data.get('time', 0)}
            self._set(f"oi_{symbol}", result)
            return result
        return None

    def get_all_open_interests(self) -> Dict[int, float]:
        result = {}
        for sym in SYMBOLS:
            oi = self.get_open_interest(sym)
            if oi:
                result[SYMBOL_TO_ID[sym]] = oi['open_interest']
            time.sleep(0.05)
        return result

    # ---- 技术指标 ----
    def get_technical_indicators(self, symbol: str, interval: str = "1d",
                                  limit: int = 200) -> Dict:
        key = f"tech_{symbol}_{interval}_{limit}"
        cached = self._cached(key, ttl=120)
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

        bb_std = float(np.std(closes[-20:])) if len(closes) >= 20 else 0
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std

        trs = [max(highs[-i]-lows[-i], abs(highs[-i]-closes[-i-1]), abs(lows[-i]-closes[-i-1]))
               for i in range(1, min(15, len(klines)))]
        atr = float(np.mean(trs)) if trs else 0

        rsi = 50.0
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_g, avg_l = np.mean(gains), np.mean(losses)
            if avg_l > 0: rsi = float(100 - 100/(1 + avg_g/avg_l))

        high_20d = float(np.max(highs[-20:])) if len(highs) >= 20 else current
        low_20d = float(np.min(lows[-20:])) if len(lows) >= 20 else current

        if len(closes) >= 200:
            pct = float(sum(1 for c in closes[-200:] if c <= current) / 200)
        else:
            pct = 0.5

        if ma20 > ma50 > ma200: alignment = 'bullish'
        elif ma20 < ma50 < ma200: alignment = 'bearish'
        else: alignment = 'mixed'

        result = {
            'price': current, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': ma20,
            'atr': atr, 'atr_pct': round(atr/current*100, 3) if current else 0,
            'rsi14': round(rsi, 1), 'high_20d': high_20d, 'low_20d': low_20d,
            'price_percentile_200d': round(pct, 4), 'ma_alignment': alignment,
        }
        self._set(key, result)
        return result

    # ---- 批量数据（SANN管线使用） ----
    def get_all_klines_df(self, interval: str = "1d", limit: int = 200) -> Dict[int, 'pd.DataFrame']:
        import pandas as pd
        result = {}
        for vid, sym in VARIETY_CODES.items():
            klines = self.get_klines(sym, interval=interval, limit=limit)
            if klines:
                df = pd.DataFrame(klines)
                df['date_col'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.sort_values('date_col').reset_index(drop=True)
                result[vid] = df
            time.sleep(0.05)
        return result


# ============================================================
def get_klines_for_technical(symbol: str, count: int = 200) -> List[Dict]:
    with BinanceDataProvider() as p:
        return p.get_klines(symbol, interval="1d", limit=count)


# ============================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    print("=" * 60)
    print("BinanceDataProvider — TradFi永续自检 (纯requests)")
    print(f"品种总数: {NUM_VARIETIES}")
    print("=" * 60)

    with BinanceDataProvider() as p:
        # 全品种价格
        prices = p.get_all_prices()
        print(f"\n📊 获取到 {len(prices)}/{NUM_VARIETIES} 品种价格:")
        for sym, price in sorted(prices.items(), key=lambda x: x[1], reverse=True)[:12]:
            print(f"   {sym:14s}: ${price:,.2f}")

        # NVDA详情
        print(f"\n📈 NVDAUSDT 详细:")
        fr = p.get_funding_rate("NVDAUSDT")
        oi = p.get_open_interest("NVDAUSDT")
        tech = p.get_technical_indicators("NVDAUSDT")
        if fr: print(f"   资金费率: {fr['funding_rate']*100:.4f}%")
        if oi: print(f"   OI: {oi['open_interest']:,.0f}")
        if tech['price']:
            print(f"   价格: ${tech['price']:.2f}")
            print(f"   MA20=${tech['ma20']:.2f} MA50=${tech['ma50']:.2f} RSI={tech['rsi14']:.0f}")
            print(f"   布林: [${tech['bb_lower']:.2f}, ${tech['bb_upper']:.2f}] 排列={tech['ma_alignment']}")

        # SPY详情
        print(f"\n📈 SPYUSDT 详细:")
        fr2 = p.get_funding_rate("SPYUSDT")
        oi2 = p.get_open_interest("SPYUSDT")
        tech2 = p.get_technical_indicators("SPYUSDT")
        if fr2: print(f"   资金费率: {fr2['funding_rate']*100:.4f}%")
        if oi2: print(f"   OI: {oi2['open_interest']:,.0f}")
        if tech2['price']:
            print(f"   价格: ${tech2['price']:.2f}")
            print(f"   MA20=${tech2['ma20']:.2f} MA50=${tech2['ma50']:.2f} RSI={tech2['rsi14']:.0f}")

        # 技术面Top信号
        print(f"\n🏆 技术面最强信号:")
        signals = []
        for sym in ["NVDAUSDT","AAPLUSDT","MSFTUSDT","TSLAUSDT","SPYUSDT","QQQUSDT","XAUUSDT","METAUSDT","GOOGLUSDT","AMZNUSDT"]:
            t = p.get_technical_indicators(sym)
            if t['price']:
                deviation = (t['price'] - t['ma20']) / t['ma20'] * 100
                signals.append((sym, t['price'], t['rsi14'], deviation, t['ma_alignment']))
        for sym, price, rsi, dev, align in sorted(signals, key=lambda x: x[2]):
            emoji = "🔴" if rsi < 35 else ("🟢" if rsi > 65 else "🟡")
            print(f"   {sym:12s}: ${price:,.2f}  RSI={rsi:.0f}  DevMA20={dev:+.1f}%  {align} {emoji}")

    print(f"\n🎉 自检完成!")
