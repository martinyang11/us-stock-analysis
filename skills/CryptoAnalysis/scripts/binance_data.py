#!/usr/bin/env python3
"""
BinanceDataProvider — Binance USDT-M 永续合约数据接口
替代原 TQSDK 数据源，为 CryptoAnalysis 和 CANN 提供统一数据层。

支持的接口：
- K线（OHLCV）：get_klines()
- 实时价格：get_price()
- 资金费率：get_funding_rate(), get_funding_rate_history()
- 持仓量OI：get_open_interest(), get_open_interest_history()
- 多空比：get_long_short_ratio()
- 技术指标：get_technical_indicators()
- 交割合约价差：get_term_structure()

前置依赖：pip install python-binance
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger('BinanceData')

# 延迟导入 Binance client（允许在无 binance 包时优雅降级）
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
                raise ImportError(
                    "需要安装 python-binance 或 binance-futures："
                    "pip install python-binance"
                )
    return _client_class


# ============================================================
# 50币种元数据（与 CANN/data/crypto_meta.json 保持一致）
# ============================================================

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "AVAXUSDT", "DOTUSDT", "LTCUSDT", "ATOMUSDT",
    "NEARUSDT", "APTUSDT", "SUIUSDT", "ICPUSDT", "SEIUSDT",
    "ETCUSDT", "MATICUSDT", "OPUSDT", "ARBUSDT", "STXUSDT",
    "STRKUSDT", "ZKUSDT", "MOVEUSDT", "UNIUSDT", "LINKUSDT",
    "MKRUSDT", "AAVEUSDT", "INJUSDT", "ENAUSDT", "JUPUSDT",
    "CRVUSDT", "LDOUSDT", "ONDOUSDT", "PENDLEUSDT",
    "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "WIFUSDT",
    "BONKUSDT", "FLOKIUSDT", "BRETTUSDT",
    "RENDERUSDT", "TAOUSDT", "FETUSDT", "WLDUSDT",
    "TIAUSDT", "PYTHUSDT", "FILUSDT", "GRTUSDT", "WUSDT",
]

VARIETY_NAMES = {i: s.replace("USDT", "") for i, s in enumerate(SYMBOLS)}
VARIETY_CODES = {i: s for i, s in enumerate(SYMBOLS)}
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
NUM_VARIETIES = 50


def get_variety_info(vid: int) -> Tuple[str, str, str]:
    """返回 (名称, 代码, 类别)"""
    from skills.CANN.data import crypto_meta_json
    # 简化实现：从json或内置映射获取
    categories = {
        **{i: "L1" for i in range(16)},
        **{i: "L2" for i in range(16, 23)},
        **{i: "DeFi" for i in range(23, 34)},
        **{i: "Meme" for i in range(34, 41)},
        **{i: "AI" for i in range(41, 45)},
        **{i: "Infra" for i in range(45, 50)},
    }
    code = VARIETY_CODES.get(vid, "")
    name = VARIETY_NAMES.get(vid, "")
    cat = categories.get(vid, "Other")
    return name, code, cat


# ============================================================
# BinanceDataProvider
# ============================================================

class BinanceDataProvider:
    """Binance USDT-M 永续合约统一数据接口

    用法：
        with BinanceDataProvider() as provider:
            klines = provider.get_klines("BTCUSDT", interval="1h", limit=200)
            funding = provider.get_funding_rate("BTCUSDT")
            oi = provider.get_open_interest("BTCUSDT")
            indicators = provider.get_technical_indicators("BTCUSDT")
    """

    def __init__(self, api_key: str = None, api_secret: str = None, use_testnet: bool = False):
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET', '')
        self.use_testnet = use_testnet
        self._client = None
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._default_cache_ttl = 300  # 5分钟默认缓存

    def _get_cache(self, key: str, ttl: float = None) -> Optional[Any]:
        """读取缓存（TTL未过期时有效）"""
        ttl = ttl or self._default_cache_ttl
        if key in self._cache and key in self._cache_ttl:
            if time.time() - self._cache_ttl[key] < ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ttl[key] = time.time()

    def _get_client(self):
        """懒加载 Binance client"""
        if self._client is None:
            Client = _get_client_class()
            if self.api_key and self.api_secret:
                self._client = Client(self.api_key, self.api_secret)
            else:
                # 公开接口不需要 API Key
                self._client = Client()
            logger.debug("Binance client 已初始化")
        return self._client

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self._client:
            try:
                self._client.close_connection()
            except Exception:
                pass
            self._client = None

    # ============================================================
    # K线数据
    # ============================================================

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 200, start_time: int = None,
                   end_time: int = None) -> List[Dict]:
        """获取K线数据

        Args:
            symbol: 交易对（如 BTCUSDT）
            interval: K线周期 1m/5m/15m/1h/4h/1d/1w
            limit: 最多返回条数（最大1500）
            start_time: 起始时间戳(ms)
            end_time: 结束时间戳(ms)

        Returns:
            [{
                'open_time': 毫秒时间戳,
                'open': float, 'high': float, 'low': float, 'close': float,
                'volume': float,  # USDT成交量
                'close_time': 毫秒时间戳,
                'quote_volume': float,
                'trades': int,
            }, ...]
        """
        cache_key = f"klines_{symbol}_{interval}_{limit}_{start_time}_{end_time}"
        cached = self._get_cache(cache_key, ttl=60)  # K线缓存60秒
        if cached:
            return cached

        client = self._get_client()
        try:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if start_time:
                params["startTime"] = start_time
            if end_time:
                params["endTime"] = end_time

            raw = client.futures_klines(**params)

            klines = []
            for k in raw:
                klines.append({
                    'open_time': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                    'close_time': k[6],
                    'quote_volume': float(k[7]),
                    'trades': k[8],
                })

            self._set_cache(cache_key, klines)
            return klines

        except Exception as e:
            logger.error(f"获取K线失败 {symbol} {interval}: {e}")
            return []

    def get_klines_df(self, symbol: str, interval: str = "1d",
                      limit: int = 500) -> 'pd.DataFrame':
        """获取K线并返回 pandas DataFrame（含date_col列）"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("需要安装 pandas：pip install pandas")

        klines = self.get_klines(symbol, interval, limit)
        if not klines:
            return pd.DataFrame()

        df = pd.DataFrame(klines)
        df['date_col'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.sort_values('date_col', ascending=True).reset_index(drop=True)
        return df

    # ============================================================
    # 实时价格
    # ============================================================

    def get_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        cache_key = f"price_{symbol}"
        cached = self._get_cache(cache_key, ttl=10)  # 价格缓存10秒
        if cached:
            return cached

        client = self._get_client()
        try:
            ticker = client.futures_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])
            self._set_cache(cache_key, price)
            return price
        except Exception as e:
            logger.error(f"获取价格失败 {symbol}: {e}")
            return None

    def get_prices_batch(self, symbols: List[str] = None) -> Dict[str, float]:
        """批量获取价格"""
        if symbols is None:
            symbols = SYMBOLS

        client = self._get_client()
        try:
            tickers = client.futures_symbol_ticker()
            result = {}
            for t in tickers:
                sym = t['symbol']
                if sym in symbols:
                    result[sym] = float(t['price'])
            return result
        except Exception as e:
            logger.error(f"批量获取价格失败: {e}")
            return {}

    # ============================================================
    # 资金费率
    # ============================================================

    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """获取最新资金费率

        Returns:
            {
                'symbol': 'BTCUSDT',
                'funding_rate': 0.0001,  # 如0.01%
                'funding_time': 毫秒时间戳（下次结算时间）,
                'mark_price': float,
            }
        """
        cache_key = f"funding_{symbol}"
        cached = self._get_cache(cache_key, ttl=120)  # 费率缓存120秒
        if cached:
            return cached

        client = self._get_client()
        try:
            raw = client.futures_funding_rate(symbol=symbol, limit=1)
            if raw:
                data = {
                    'symbol': raw[0]['symbol'],
                    'funding_rate': float(raw[0]['fundingRate']),
                    'funding_time': raw[0].get('fundingTime', 0),
                    'mark_price': float(raw[0].get('markPrice', 0)),
                }
                self._set_cache(cache_key, data)
                return data
        except Exception as e:
            logger.error(f"获取资金费率失败 {symbol}: {e}")
        return None

    def get_funding_rate_history(self, symbol: str, limit: int = 100) -> List[Dict]:
        """获取历史资金费率"""
        cache_key = f"funding_hist_{symbol}_{limit}"
        cached = self._get_cache(cache_key, ttl=600)  # 历史费率缓存10分钟
        if cached:
            return cached

        client = self._get_client()
        try:
            raw = client.futures_funding_rate(symbol=symbol, limit=limit)
            result = []
            for r in raw:
                result.append({
                    'symbol': r['symbol'],
                    'funding_rate': float(r['fundingRate']),
                    'funding_time': r.get('fundingTime', 0),
                    'mark_price': float(r.get('markPrice', 0)),
                })
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"获取历史资金费率失败 {symbol}: {e}")
            return []

    def get_funding_rate_percentile(self, symbol: str, window: int = 100) -> Tuple[float, float]:
        """获取当前资金费率的历史分位数

        Returns:
            (current_rate, percentile) — percentile ∈ [0, 100]
        """
        current = self.get_funding_rate(symbol)
        if not current:
            return 0.0, 50.0

        history = self.get_funding_rate_history(symbol, limit=window)
        if len(history) < 10:
            return current['funding_rate'], 50.0

        rates = sorted([h['funding_rate'] for h in history])
        current_rate = current['funding_rate']

        # 计算分位数
        n_lower = sum(1 for r in rates if r <= current_rate)
        percentile = (n_lower / len(rates)) * 100

        return current_rate, percentile

    def get_all_funding_rates(self) -> Dict[str, Dict]:
        """批量获取所有币种的资金费率"""
        client = self._get_client()
        try:
            raw = client.futures_funding_rate(limit=500)
            result = {}
            for r in raw:
                sym = r['symbol']
                if sym in SYMBOLS:
                    result[sym] = {
                        'funding_rate': float(r['fundingRate']),
                        'funding_time': r.get('fundingTime', 0),
                        'mark_price': float(r.get('markPrice', 0)),
                    }
            return result
        except Exception as e:
            logger.error(f"批量获取资金费率失败: {e}")
            return {}

    # ============================================================
    # 持仓量 OI
    # ============================================================

    def get_open_interest(self, symbol: str) -> Optional[Dict]:
        """获取当前未平仓合约

        Returns:
            {'symbol': 'BTCUSDT', 'open_interest': 12345.67, 'time': 毫秒}
        """
        cache_key = f"oi_{symbol}"
        cached = self._get_cache(cache_key, ttl=120)
        if cached:
            return cached

        client = self._get_client()
        try:
            raw = client.futures_open_interest(symbol=symbol)
            data = {
                'symbol': raw['symbol'],
                'open_interest': float(raw['openInterest']),
                'time': raw.get('time', 0),
            }
            self._set_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"获取OI失败 {symbol}: {e}")
            return None

    def get_open_interest_history(self, symbol: str,
                                   period: str = "1d",
                                   limit: int = 30) -> List[Dict]:
        """获取历史OI（需要API Key）

        Args:
            period: 5m/15m/30m/1h/2h/4h/6h/12h/1d
            limit: 最多30
        """
        if not self.api_key:
            logger.warning("OI历史数据需要 API Key，回退到从K线估算")
            return self._estimate_oi_from_volume(symbol, limit)

        cache_key = f"oi_hist_{symbol}_{period}_{limit}"
        cached = self._get_cache(cache_key, ttl=600)
        if cached:
            return cached

        client = self._get_client()
        try:
            raw = client.futures_open_interest_hist(
                symbol=symbol, period=period, limit=limit
            )
            result = []
            for r in raw:
                result.append({
                    'symbol': r['symbol'],
                    'open_interest': float(r['sumOpenInterest']),
                    'open_interest_value': float(r.get('sumOpenInterestValue', 0)),
                    'timestamp': r['timestamp'],
                })
            self._set_cache(cache_key, result)
            return result
        except Exception as e:
            logger.warning(f"获取OI历史失败 {symbol}: {e}，回退到从K线估算")
            return self._estimate_oi_from_volume(symbol, limit)

    def _estimate_oi_from_volume(self, symbol: str, count: int = 30) -> List[Dict]:
        """从K线成交量估算OI趋势（降级方案）"""
        # 获取当前OI
        current_oi = self.get_open_interest(symbol)
        current_oi_val = current_oi['open_interest'] if current_oi else 0

        # 用K线成交量加权估算OI变化
        klines = self.get_klines(symbol, interval="1d", limit=count)
        if not klines:
            return []

        result = []
        for i, k in enumerate(klines):
            # 简单估算：OI ≈ 当前OI * (volume_ratio)
            vol_ratio = k['volume'] / (klines[-1]['volume'] or 1) if klines[-1]['volume'] else 1
            estimated_oi = current_oi_val * max(0.3, min(1.5, vol_ratio))
            result.append({
                'symbol': symbol,
                'open_interest': estimated_oi,
                'open_interest_value': estimated_oi * k['close'],
                'timestamp': k['open_time'],
                '_estimated': True,
            })
        return result

    # ============================================================
    # 多空比
    # ============================================================

    def get_long_short_ratio(self, symbol: str, period: str = "5m",
                              limit: int = 1) -> Optional[Dict]:
        """获取多空持仓比（需要API Key）

        Returns:
            {'long_ratio': 0.55, 'short_ratio': 0.45, 'ls_ratio': 1.22}
        """
        if not self.api_key:
            return None

        cache_key = f"lsr_{symbol}_{period}"
        cached = self._get_cache(cache_key, ttl=300)
        if cached:
            return cached

        client = self._get_client()
        try:
            raw = client.futures_long_short_ratio(
                symbol=symbol, period=period, limit=limit
            )
            if raw:
                r = raw[0]
                ls_ratio = float(r.get('longShortRatio', 1.0))
                long_pct = ls_ratio / (1 + ls_ratio)
                short_pct = 1 - long_pct
                data = {
                    'long_ratio': round(long_pct, 4),
                    'short_ratio': round(short_pct, 4),
                    'ls_ratio': round(ls_ratio, 4),
                    'timestamp': r.get('timestamp', 0),
                }
                self._set_cache(cache_key, data)
                return data
        except Exception as e:
            logger.warning(f"获取多空比失败 {symbol}: {e}")
        return None

    # ============================================================
    # 交割合约（期限结构）
    # ============================================================

    def get_delivery_prices(self, symbol: str) -> Dict[str, float]:
        """获取交割合约价格（用于计算期限结构）

        仅 BTC/ETH 有流动性好的交割合约。

        Returns:
            {'this_week': 价格, 'next_week': 价格, 'this_quarter': 价格, 'next_quarter': 价格}
        """
        cache_key = f"delivery_{symbol}"
        cached = self._get_cache(cache_key, ttl=300)
        if cached:
            return cached

        client = self._get_client()
        try:
            # 获取所有交割合约
            exchange_info = client.futures_exchange_info()
            base = symbol.replace("USDT", "")

            delivery_pairs = {}
            # Binance 交割合约格式: BTCUSDT_250627 (YYMMDD)
            import re
            for s in exchange_info['symbols']:
                if s['symbol'].startswith(f"{base}USD") and s['contractType'] != 'PERPETUAL':
                    delivery_pairs[s['contractType']] = s['symbol']

            result = {}
            for ctype, dsym in delivery_pairs.items():
                try:
                    ticker = client.futures_symbol_ticker(symbol=dsym)
                    result[ctype.lower().replace('_', '_')] = float(ticker['price'])
                except Exception:
                    pass

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.warning(f"获取交割合约价格失败 {symbol}: {e}")
            return {}

    def get_term_structure(self, symbol: str) -> Dict:
        """计算期限结构

        Returns:
            {
                'perpetual_price': float,
                'delivery_prices': {...},
                'contango_pct': float,  # 季-当周升水率
                'structure': 'contango' | 'backwardation' | 'flat' | 'unavailable'
            }
        """
        perp_price = self.get_price(symbol)
        delivery = self.get_delivery_prices(symbol)

        result = {
            'perpetual_price': perp_price,
            'delivery_prices': delivery,
            'contango_pct': 0.0,
            'structure': 'unavailable',
        }

        if not perp_price or not delivery:
            return result

        # 计算近远月升水率
        prices = list(delivery.values())
        if len(prices) >= 2:
            near = prices[0]
            far = prices[-1]
            contango = (far - near) / near * 100
            result['contango_pct'] = round(contango, 4)

            if contango > 0.5:
                result['structure'] = 'contango'
            elif contango < -0.5:
                result['structure'] = 'backwardation'
            else:
                result['structure'] = 'flat'

        return result

    # ============================================================
    # 技术指标
    # ============================================================

    def get_technical_indicators(self, symbol: str, interval: str = "1d",
                                  limit: int = 200) -> Dict:
        """计算技术指标

        Returns:
            {
                'price': float,
                'ma5': float, 'ma10': float, 'ma20': float, 'ma50': float, 'ma200': float,
                'bb_upper': float, 'bb_middle': float, 'bb_lower': float,
                'atr': float, 'atr_pct': float,
                'rsi14': float,
                'high_20d': float, 'low_20d': float,
                'price_percentile_200d': float,
                'ma_alignment': 'bullish' | 'bearish' | 'mixed',
            }
        """
        cache_key = f"tech_{symbol}_{interval}_{limit}"
        cached = self._get_cache(cache_key, ttl=120)
        if cached:
            return cached

        klines = self.get_klines(symbol, interval=interval, limit=limit)
        if len(klines) < 50:
            return self._empty_tech_indicators()

        closes = np.array([k['close'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])

        current_price = closes[-1]

        # 均线
        def ma(data, period):
            if len(data) < period:
                return float(data[-1])
            return float(np.mean(data[-period:]))

        ma5 = ma(closes, 5)
        ma10 = ma(closes, 10)
        ma20 = ma(closes, 20)
        ma50 = ma(closes, 50) if len(closes) >= 50 else ma20
        ma200 = ma(closes, 200) if len(closes) >= 200 else ma50

        # 布林带（20日均线 ± 2σ）
        bb_mid = ma20
        bb_std = float(np.std(closes[-20:])) if len(closes) >= 20 else 0
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # ATR（14周期）
        tr_list = []
        for i in range(1, min(15, len(klines))):
            h, l, pc = highs[-i], lows[-i], closes[-i-1]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            tr_list.append(tr)
        atr = float(np.mean(tr_list)) if tr_list else 0
        atr_pct = (atr / current_price * 100) if current_price > 0 else 0

        # RSI(14)
        rsi14 = self._calc_rsi(closes, 14)

        # 20日高低
        high_20d = float(np.max(highs[-20:])) if len(highs) >= 20 else current_price
        low_20d = float(np.min(lows[-20:])) if len(lows) >= 20 else current_price

        # 200日价格分位数
        if len(closes) >= 200:
            n_lower = sum(1 for c in closes[-200:] if c <= current_price)
            price_percentile = float(n_lower / 200)
        elif len(closes) >= 50:
            n_lower = sum(1 for c in closes[-50:] if c <= current_price)
            price_percentile = float(n_lower / len(closes[-50:]))
        else:
            price_percentile = 0.5

        # 均线排列
        if ma20 > ma50 > ma200:
            alignment = 'bullish'
        elif ma20 < ma50 < ma200:
            alignment = 'bearish'
        else:
            alignment = 'mixed'

        result = {
            'price': current_price,
            'ma5': ma5, 'ma10': ma10, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'bb_upper': bb_upper, 'bb_middle': bb_mid, 'bb_lower': bb_lower,
            'atr': atr, 'atr_pct': round(atr_pct, 3),
            'rsi14': rsi14,
            'high_20d': high_20d, 'low_20d': low_20d,
            'price_percentile_200d': round(price_percentile, 4),
            'ma_alignment': alignment,
        }

        self._set_cache(cache_key, result)
        return result

    def _calc_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - 100 / (1 + rs))

    def _empty_tech_indicators(self) -> Dict:
        return {
            'price': 0, 'ma5': 0, 'ma10': 0, 'ma20': 0, 'ma50': 0, 'ma200': 0,
            'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0,
            'atr': 0, 'atr_pct': 0,
            'rsi14': 50, 'high_20d': 0, 'low_20d': 0,
            'price_percentile_200d': 0.5, 'ma_alignment': 'mixed',
        }

    # ============================================================
    # 批量数据采集（CANN 管线使用）
    # ============================================================

    def get_all_klines_df(self, interval: str = "1d",
                          limit: int = 200) -> Dict[int, 'pd.DataFrame']:
        """批量获取全品种K线DataFrame

        Returns:
            {variety_id: DataFrame}
        """
        import pandas as pd

        result = {}
        for vid, symbol in VARIETY_CODES.items():
            klines = self.get_klines(symbol, interval=interval, limit=limit)
            if klines:
                df = pd.DataFrame(klines)
                df['date_col'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.sort_values('date_col').reset_index(drop=True)
                result[vid] = df
            time.sleep(0.05)  # 频率控制

        return result

    def get_all_funding_rates_dict(self) -> Dict[int, float]:
        """批量获取全品种资金费率

        Returns:
            {variety_id: funding_rate}
        """
        all_rates = self.get_all_funding_rates()
        result = {}
        for sym, data in all_rates.items():
            vid = SYMBOL_TO_ID.get(sym)
            if vid is not None:
                result[vid] = data['funding_rate']
        return result

    def get_all_open_interests(self) -> Dict[int, float]:
        """批量获取全品种OI"""
        client = self._get_client()
        try:
            raw = client.futures_open_interest(limit=500)
            result = {}
            for r in raw:
                sym = r['symbol']
                vid = SYMBOL_TO_ID.get(sym)
                if vid is not None:
                    result[vid] = float(r['openInterest'])
            return result
        except Exception as e:
            logger.error(f"批量获取OI失败: {e}")
            return {}


# ============================================================
# 便捷函数
# ============================================================

def get_klines_for_technical(symbol: str, count: int = 200) -> List[Dict]:
    """CANN 兼容接口：获取K线用于技术面计算"""
    with BinanceDataProvider() as provider:
        return provider.get_klines(symbol, interval="1d", limit=count)


def get_tech_scores(symbol: str) -> Dict:
    """获取完整技术面评分数据"""
    with BinanceDataProvider() as provider:
        return provider.get_technical_indicators(symbol)


# ============================================================
# CLI 测试入口
# ============================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("BinanceDataProvider 自检测试")
    print("=" * 60)

    with BinanceDataProvider() as provider:
        # 测试1: BTC价格
        print("\n1. BTCUSDT 价格...")
        price = provider.get_price("BTCUSDT")
        print(f"   BTC = ${price:,.2f}" if price else "   获取失败")

        # 测试2: K线
        print("\n2. BTCUSDT 日K线(最近5条)...")
        klines = provider.get_klines("BTCUSDT", interval="1d", limit=5)
        for k in klines[-3:]:
            dt = datetime.fromtimestamp(k['open_time'] / 1000).strftime('%Y-%m-%d')
            print(f"   {dt}: O={k['open']:.2f} H={k['high']:.2f} L={k['low']:.2f} C={k['close']:.2f} V={k['volume']:.0f}")

        # 测试3: 资金费率
        print("\n3. BTCUSDT 资金费率...")
        funding = provider.get_funding_rate("BTCUSDT")
        if funding:
            print(f"   费率={funding['funding_rate']*100:.4f}% 标记价=${funding['mark_price']:.2f}")

        # 测试4: 技术指标
        print("\n4. BTCUSDT 技术指标...")
        tech = provider.get_technical_indicators("BTCUSDT")
        print(f"   价格=${tech['price']:.2f} MA20=${tech['ma20']:.2f} MA50=${tech['ma50']:.2f}")
        print(f"   布林带: [${tech['bb_lower']:.2f}, ${tech['bb_upper']:.2f}]")
        print(f"   RSI(14)={tech['rsi14']:.1f} ATR={tech['atr_pct']:.2f}%")
        print(f"   均线: {tech['ma_alignment']}  200日分位={tech['price_percentile_200d']:.0%}")

        # 测试5: 期限结构（仅BTC）
        print("\n5. BTC 期限结构...")
        term = provider.get_term_structure("BTCUSDT")
        print(f"   结构={term['structure']} 升水率={term['contango_pct']:.4f}%")
        if term['delivery_prices']:
            for ctype, p in term['delivery_prices'].items():
                print(f"   {ctype}: ${p:.2f}")

        # 测试6: OI
        print("\n6. BTCUSDT OI...")
        oi = provider.get_open_interest("BTCUSDT")
        if oi:
            print(f"   OI={oi['open_interest']:,.0f}")

        # 测试7: 全品种资金费率
        print("\n7. 全品种资金费率(前10)...")
        all_rates = provider.get_all_funding_rates()
        sorted_rates = sorted(all_rates.items(), key=lambda x: x[1]['funding_rate'], reverse=True)
        for sym, data in sorted_rates[:5]:
            print(f"   {sym}: {data['funding_rate']*100:.4f}%")
        print(f"   ...共{len(all_rates)}个品种")

    print("\n" + "=" * 60)
    print("自检完成")
