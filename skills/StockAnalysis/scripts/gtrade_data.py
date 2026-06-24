#!/usr/bin/env python3
"""
GtradeDataProvider — gTrade 去中心化永续合约数据接口
为 StockAnalysis 和 SANN 提供 TradFi 品种的统一数据层。

数据源:
  - gTrade REST API: 品种元数据 / 市场状态 / spread
  - gTrade WebSocket v4: 实时 mark 价格
  - Yahoo Finance: 历史 OHLC K线 (gTrade 股价跟踪标的现货)

前置依赖: pip install requests numpy yfinance websocket-client
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger('GtradeData')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from skills.common.tradfi_universe import (
        CATEGORY_OVERRIDES,
        GTRADE_NAME_ALIASES,
        GTRADE_PAIR_INDICES,
        INDEX_SYMBOLS,
        STOCK_SYMBOLS,
        TRADFI_SYMBOLS,
        TRADFI_SYMBOL_SET,
        YF_TICKER_OVERRIDES,
    )
except Exception:
    TRADFI_SYMBOLS = []
    TRADFI_SYMBOL_SET = set()
    STOCK_SYMBOLS = []
    INDEX_SYMBOLS = []
    GTRADE_PAIR_INDICES = {}
    GTRADE_NAME_ALIASES = {}
    YF_TICKER_OVERRIDES = {}
    CATEGORY_OVERRIDES = {}

# ============================================================
# gTrade API 端点
# ============================================================
GTRADE_BACKEND = "https://backend-arbitrum.gains.trade"
GTRADE_PRICING_WS = "wss://backend-pricing.eu.gains.trade/v4"

# gTrade 品种分组 (groupIndex → 名称)
GROUP_NAMES = {
    0: "crypto",
    2: "stocks", 3: "stocks", 4: "stocks",
    5: "indices",
    6: "commodities", 7: "commodities",
}

# ============================================================
# HTTP 代理配置
# ============================================================
PROXY_URL = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or os.environ.get('ALL_PROXY') or ''
PROXIES = {'http': PROXY_URL, 'https': PROXY_URL} if PROXY_URL else None


def _get(path: str, params: dict = None, base_url: str = None,
         timeout: int = 20) -> dict:
    """封装 gTrade REST GET 请求"""
    url = f"{base_url or GTRADE_BACKEND}{path}"
    try:
        r = requests.get(url, params=params, timeout=timeout, proxies=PROXIES)
        r.raise_for_status()
        data = r.json()
        return data
    except Exception as e:
        logger.error(f"gTrade 请求失败 {path}: {e}")
        return {}


# ============================================================
# TradFi 品种列表 (gTrade pairIndex → 元数据)
# ============================================================

# 构建时从 /trading-variables/all 动态加载
_raw_pairs: List[dict] = []
_raw_groups: List[dict] = []
_tradfi_pairs: List[dict] = []  # 过滤后的 TradFi 品种
_pair_index_map: Dict[int, dict] = {}  # pairIndex → pair 元数据
_name_to_index: Dict[str, int] = {}  # "NVDA" → pairIndex
_variety_list: List[dict] = []  # SANN 兼容的品种列表


def _build_tradfi_list():
    """从 gTrade API 加载品种列表并构建 TradFi 映射"""
    global _raw_pairs, _raw_groups, _tradfi_pairs
    global _pair_index_map, _name_to_index, _variety_list

    if _tradfi_pairs:
        return  # 已加载

    data = _get("/trading-variables/all", timeout=3)
    pairs = data.get('pairs', [])
    groups = data.get('groups', [])
    if not pairs:
        logger.error("无法加载 gTrade 品种列表，使用29标的静态兜底")
        for name in TRADFI_SYMBOLS:
            pair_index = GTRADE_PAIR_INDICES.get(name, -1)
            group = 'stocks' if name in STOCK_SYMBOLS else 'indices'
            info = {
                'pairIndex': pair_index,
                'name': name,
                'symbol': f'{name}/USD',
                'group': group,
                'groupIndex': -1,
                'spreadP': 0,
                'spread_pct': 0.0,
                'feeIdx': 0,
                'maxLeverage': 0,
            }
            _tradfi_pairs.append(info)
            _pair_index_map[pair_index] = info
            _name_to_index[name] = pair_index

        for i, info in enumerate(_tradfi_pairs):
            _variety_list.append({
                'variety_id': i,
                'name': info['name'],
                'symbol': info['symbol'],
                'pairIndex': info['pairIndex'],
                'group': info['group'],
                'category': _get_category(info['name'], info['group']),
            })
        return

    _raw_pairs = pairs
    _raw_groups = groups

    seen_names = set()
    for idx, p in enumerate(pairs):
        gi = int(p.get('groupIndex', -1))
        raw_name = p.get('from', '?')
        name = GTRADE_NAME_ALIASES.get(raw_name, raw_name)
        to_ = p.get('to', 'USD')

        # 跳过 _1 重复 + FB(META alias)
        if '_1' in raw_name or raw_name == 'FB':
            continue

        if TRADFI_SYMBOL_SET and name not in TRADFI_SYMBOL_SET:
            continue

        # 只取 stocks/indices/commodities/crypto 主流
        if gi not in GROUP_NAMES:
            continue
        group = GROUP_NAMES[gi]
        if name in {'SPCX'}:
            group = 'indices'

        # 去重
        if name in seen_names:
            continue
        seen_names.add(name)

        spread_p = int(p.get('spreadP', '0'))
        spread_pct = round(spread_p / 1e9 * 100, 4)
        max_lev = int(groups[gi].get('maxLeverage', '0')) / 1000 if gi < len(groups) else 50

        info = {
            'pairIndex': idx,
            'name': name,
            'symbol': f'{name}/USD',
            'group': group,
            'groupIndex': gi,
            'spreadP': spread_p,
            'spread_pct': spread_pct,
            'feeIdx': int(p.get('feeIndex', 0)),
            'maxLeverage': max_lev,
        }
        _tradfi_pairs.append(info)
        _pair_index_map[idx] = info
        _name_to_index[name] = idx

    if TRADFI_SYMBOLS:
        order = {sym: i for i, sym in enumerate(TRADFI_SYMBOLS)}
        _tradfi_pairs.sort(key=lambda x: order.get(x['name'], len(order)))

    # 构建 SANN 兼容品种列表
    for i, info in enumerate(_tradfi_pairs):
        _variety_list.append({
            'variety_id': i,
            'name': info['name'],
            'symbol': info['symbol'],
            'pairIndex': info['pairIndex'],
            'group': info['group'],
            'category': _get_category(info['name'], info['group']),
        })

    logger.info(f"gTrade TradFi 品种加载完成: {len(_tradfi_pairs)} 个")


def _get_category(name: str, group: str) -> str:
    """品种分类"""
    if name in CATEGORY_OVERRIDES:
        return CATEGORY_OVERRIDES[name]
    if group == 'crypto':
        return '加密'
    if group == 'commodities':
        return '商品'
    if group == 'indices':
        return 'ETF/指数'

    # 股票分类
    tech_giants = {'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA'}
    semis = {'INTC', 'AMD'}
    crypto_adj = {'MSTR', 'COIN', 'CRCL', 'HOOD', 'MARA', 'RIOT'}
    fintech = {'V', 'MA', 'PYPL', 'SOFI'}
    consumer = {'DIS', 'NKE', 'KO', 'MCD', 'WMT', 'SBUX', 'ABNB', 'GME'}
    pharma = {'PFE'}
    enterprise = {'BA', 'LMT'}
    tech_other = {'SNAP', 'NFLX', 'PLTR', 'ROKU', 'BIDU', 'SBET', 'WPM'}

    if name in tech_giants:
        return '科技巨头'
    if name in semis:
        return '半导体'
    if name in crypto_adj:
        return '加密相关'
    if name in fintech:
        return '金融科技'
    if name in consumer:
        return '消费零售'
    if name in pharma:
        return '医药'
    if name in enterprise:
        return '工业/国防'
    if name in tech_other:
        return '科技其他'
    return '股票'


def get_tradfi_pairs() -> List[dict]:
    """获取 gTrade TradFi 品种列表"""
    _build_tradfi_list()
    return _tradfi_pairs


def get_variety_list() -> List[dict]:
    """获取 SANN 兼容品种列表"""
    _build_tradfi_list()
    return _variety_list


def get_pair_index_by_name(name: str) -> Optional[int]:
    """通过品种名获取 gTrade pairIndex"""
    _build_tradfi_list()
    return _name_to_index.get(name.upper())


def get_pair_info(pair_index: int) -> Optional[dict]:
    """通过 pairIndex 获取品种元数据"""
    _build_tradfi_list()
    return _pair_index_map.get(pair_index)


NUM_VARIETIES = property(lambda self: len(_variety_list))


# ============================================================
# yfinance 品种名映射 (gTrade name → Yahoo Finance ticker)
# ============================================================
YF_MAP = {
    # gTrade name → Yahoo Finance ticker
    'BTC': 'BTC-USD', 'ETH': 'ETH-USD',
    'XAU': 'GC=F',   # Gold futures
    'XAG': 'SI=F',   # Silver futures
    'WTI': 'CL=F',   # Crude oil futures
    'BRENT': 'BZ=F', # Brent crude
    'NATGAS': 'NG=F', # Natural gas
    'HG': 'HG=F',    # Copper
    'XPT': 'PL=F',   # Platinum
    'XPD': 'PA=F',   # Palladium
    'SPX500': '^GSPC', 'SPCX': '^GSPC', 'NAS100': '^NDX', 'USA30': '^DJI',
    # Stocks: same name on Yahoo
    # Fallback: f'{name}' (most stocks work directly)
}


def _to_yf_ticker(name: str) -> str:
    """gTrade 品种名 → Yahoo Finance ticker"""
    if name in YF_TICKER_OVERRIDES:
        return YF_TICKER_OVERRIDES[name]
    if name in YF_MAP:
        return YF_MAP[name]
    # 默认直接用品种名 (大多数美股直接用 ticker)
    return name


# ============================================================
class GtradeDataProvider:
    """gTrade 数据接口 — REST + WebSocket + Yahoo Finance"""

    def __init__(self, use_ws: bool = True):
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._ws = None
        self._ws_thread = None
        self._ws_running = False
        self._prices: Dict[int, float] = {}  # pairIndex → mark价格
        self._use_ws = use_ws

        # 确保品种列表已加载
        _build_tradfi_list()

    def _cached(self, key: str, ttl: float = 300) -> Optional[Any]:
        if key in self._cache and key in self._cache_ttl:
            if time.time() - self._cache_ttl[key] < ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any):
        self._cache[key] = value
        self._cache_ttl[key] = time.time()

    def __enter__(self):
        if self._use_ws:
            self.connect_ws()
        return self

    def __exit__(self, *args):
        self.disconnect_ws()

    # ---- WebSocket v4 ----
    def connect_ws(self):
        """连接 gTrade v4 WebSocket 实时价格流"""
        if self._ws_running:
            return

        def _ws_loop():
            import websocket
            while self._ws_running:
                try:
                    ws_url = GTRADE_PRICING_WS
                    # websocket-client 代理支持
                    ws_kwargs = {}
                    if PROXY_URL:
                        # 解析代理 URL
                        from urllib.parse import urlparse
                        parsed = urlparse(PROXY_URL)
                        ws_kwargs['http_proxy_host'] = parsed.hostname
                        ws_kwargs['http_proxy_port'] = parsed.port or 7897
                        if parsed.username:
                            ws_kwargs['http_proxy_auth'] = (parsed.username, parsed.password or '')

                    ws = websocket.create_connection(ws_url, **ws_kwargs)
                    logger.info("gTrade WebSocket v4 已连接")
                    while self._ws_running:
                        msg = ws.recv()
                        data = json.loads(msg)
                        # v4 格式: {"m": [pairIdx, price, ...], "i": [...], "t": ts}
                        if 'm' in data:
                            m = data['m']
                            for i in range(0, len(m), 2):
                                pidx = int(m[i])
                                price = float(m[i + 1])
                                self._prices[pidx] = price
                        if 't' in data:
                            self._last_ws_ts = data['t']
                    ws.close()
                except Exception as e:
                    logger.warning(f"WebSocket 断开: {e}, 3秒后重连...")
                    time.sleep(3)

        self._ws_running = True
        self._ws_thread = threading.Thread(target=_ws_loop, daemon=True)
        self._ws_thread.start()
        # 等待首批数据
        for _ in range(30):  # 最多等3秒
            if self._prices:
                break
            time.sleep(0.1)
        logger.info(f"WebSocket 首批价格: {len(self._prices)} 品种")

    def disconnect_ws(self):
        self._ws_running = False
        if self._ws_thread:
            self._ws_thread.join(timeout=2)
            self._ws_thread = None

    # ---- 价格 ----
    def get_price(self, pair_index: int) -> Optional[float]:
        """获取 gTrade mark 价格（WebSocket 优先，降级到 yfinance）"""
        if pair_index in self._prices and self._prices[pair_index] > 0:
            return self._prices[pair_index]

        # 降级: yfinance
        info = _pair_index_map.get(pair_index)
        if info:
            try:
                ticker = _to_yf_ticker(info['name'])
                yt = yf.Ticker(ticker)
                hist = yt.history(period='1d')
                if len(hist) > 0:
                    return float(hist['Close'].iloc[-1])
            except Exception:
                pass
        return None

    def get_price_by_name(self, name: str) -> Optional[float]:
        pidx = get_pair_index_by_name(name)
        if pidx is not None:
            return self.get_price(pidx)
        return None

    # ---- K线 (yfinance) ----
    def get_klines(self, name: str, interval: str = "1d",
                   limit: int = 200) -> List[Dict]:
        """获取历史 K线 (Yahoo Finance)

        Args:
            name: 品种名 (如 "NVDA")
            interval: 1d, 1h, 1wk 等
            limit: 返回条数

        Returns:
            [{"open": ..., "high": ..., "low": ..., "close": ...,
              "volume": ..., "open_time": ...}, ...]
        """
        cache_key = f"kl_{name}_{interval}_{limit}"
        cached = self._cached(cache_key, ttl=120)
        if cached:
            return cached

        try:
            ticker = _to_yf_ticker(name)
            period_map = {
                '1d': f'{limit}d', '1h': f'{min(limit, 730)}d',
                '1wk': f'{limit * 7}d', '1mo': f'{limit * 30}d',
            }
            period = period_map.get(interval, f'{limit}d')

            yt = yf.Ticker(ticker)
            df = yt.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"yfinance 无数据: {ticker}")
                return []

            klines = []
            for idx, row in df.iterrows():
                klines.append({
                    'open_time': int(idx.timestamp() * 1000),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']),
                    'close_time': int(idx.timestamp() * 1000),
                    'quote_volume': float(row['Volume']),
                    'trades': 0,
                })

            # 取最后 limit 条
            result = klines[-limit:] if len(klines) > limit else klines
            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"yfinance K线失败 {name}: {e}")
            return []

    # ---- 技术指标 ----
    def get_technical_indicators(self, name: str, interval: str = "1d",
                                  limit: int = 200) -> Dict:
        """计算技术指标（从 yfinance K线）"""
        cache_key = f"tech_{name}_{interval}_{limit}"
        cached = self._cached(cache_key, ttl=120)
        if cached:
            return cached

        klines = self.get_klines(name, interval=interval, limit=limit)

        if len(klines) < 20:
            return {
                'price': 0, 'ma20': 0, 'ma50': 0, 'ma200': 0,
                'bb_upper': 0, 'bb_lower': 0, 'atr': 0, 'atr_pct': 0,
                'rsi14': 50, 'high_20d': 0, 'low_20d': 0,
                'price_percentile_200d': 0.5, 'ma_alignment': 'mixed',
            }

        closes = np.array([k['close'] for k in klines])
        highs = np.array([k['high'] for k in klines])
        lows = np.array([k['low'] for k in klines])
        current = closes[-1]

        def ma(d, p):
            return float(np.mean(d[-p:])) if len(d) >= p else float(d[-1])

        ma20, ma50 = ma(closes, 20), ma(closes, 50)
        ma200 = ma(closes, 200) if len(closes) >= 200 else ma50

        bb_std = float(np.std(closes[-20:])) if len(closes) >= 20 else 0
        bb_upper = ma20 + 2 * bb_std
        bb_lower = ma20 - 2 * bb_std

        # ATR
        trs = []
        for i in range(1, min(15, len(klines))):
            tr = max(
                highs[-i] - lows[-i],
                abs(highs[-i] - closes[-i - 1]),
                abs(lows[-i] - closes[-i - 1])
            )
            trs.append(tr)
        atr = float(np.mean(trs)) if trs else 0

        # RSI14
        rsi = 50.0
        if len(closes) >= 15:
            deltas = np.diff(closes[-15:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_g, avg_l = np.mean(gains), np.mean(losses)
            if avg_l > 0:
                rsi = float(100 - 100 / (1 + avg_g / avg_l))

        high_20d = float(np.max(highs[-20:])) if len(highs) >= 20 else current
        low_20d = float(np.min(lows[-20:])) if len(lows) >= 20 else current

        if len(closes) >= 200:
            pct = float(sum(1 for c in closes[-200:] if c <= current) / 200)
        else:
            pct = 0.5

        if ma20 > ma50 > ma200:
            alignment = 'bullish'
        elif ma20 < ma50 < ma200:
            alignment = 'bearish'
        else:
            alignment = 'mixed'

        result = {
            'price': current, 'ma20': ma20, 'ma50': ma50, 'ma200': ma200,
            'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': ma20,
            'atr': atr, 'atr_pct': round(atr / current * 100, 3) if current else 0,
            'rsi14': round(rsi, 1), 'high_20d': high_20d, 'low_20d': low_20d,
            'price_percentile_200d': round(pct, 4), 'ma_alignment': alignment,
        }
        self._set_cache(cache_key, result)
        return result

    # ---- gTrade 特有 ----
    def get_spread(self, name: str) -> Optional[float]:
        """获取 gTrade spread (%)"""
        pidx = get_pair_index_by_name(name)
        info = _pair_index_map.get(pidx) if pidx is not None else None
        return info['spread_pct'] if info else None

    def is_stocks_open(self) -> bool:
        """美股当前是否开盘"""
        cached = self._cached('market_status', ttl=60)
        if cached is not None:
            return cached

        data = _get("/trading-variables/all")
        status = data.get('isStocksOpen', False)
        self._set_cache('market_status', status)
        return status

    def get_market_status(self) -> Dict[str, bool]:
        """获取所有市场开关状态"""
        cached = self._cached('full_market_status', ttl=60)
        if cached:
            return cached

        data = _get("/trading-variables/all")
        status = {
            'stocks': data.get('isStocksOpen', False),
            'forex': data.get('isForexOpen', False),
            'commodities': data.get('isCommoditiesOpen', False),
            'indices': data.get('isIndicesOpen', False),
            'crypto': data.get('isCryptoOpen', True),  # 加密永远开盘
        }
        self._set_cache('full_market_status', status)
        return status

    def get_24h_volume(self, name: str) -> Optional[float]:
        """获取24h交易量 (yfinance)"""
        try:
            ticker = _to_yf_ticker(name)
            yt = yf.Ticker(ticker)
            hist = yt.history(period='1d')
            if len(hist) > 0:
                return float(hist['Volume'].iloc[-1])
        except Exception:
            pass
        return None

    # ---- 批量 ----
    def get_all_klines_df(self, interval: str = "1d", limit: int = 200) -> Dict[int, 'pd.DataFrame']:
        """批量获取K线 (SANN 管线兼容接口)"""
        import pandas as pd
        result = {}
        for vid, info in enumerate(_variety_list):
            klines = self.get_klines(info['name'], interval=interval, limit=limit)
            if klines:
                df = pd.DataFrame(klines)
                df['date_col'] = pd.to_datetime(df['open_time'], unit='ms')
                df = df.sort_values('date_col').reset_index(drop=True)
                result[vid] = df
            time.sleep(0.05)
        return result


# ============================================================
# 24维技术面评分 (SANN 管线用)
# ============================================================
def compute_technical_scores(df: 'pd.DataFrame') -> List[float]:
    """从 OHLC DataFrame 计算 24 维 [0,1] 技术面评分

    T1-T3:   均线偏离 (MA20/50/200)
    T4:      MA排列
    T5-T7:   布林带 (上轨偏离/下轨偏离/带宽)
    T8:      RSI14
    T9:      ATR%
    T10-T13: 动量 (1d/5d/10d/20d)
    T14-T17: 量比 (1d/5d/10d/20d)
    T18-T21: 距高低位 (20d/50d)
    T22-T24: 波动率分位 (20d/50d/200d)
    """
    import pandas as pd
    closes = df['close'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    volumes = df['volume'].values.astype(float) if 'volume' in df.columns else np.ones_like(closes)

    n = len(closes)
    if n < 20:
        return [0.5] * 24

    current = closes[-1]

    def ma(arr, p):
        return float(np.mean(arr[-p:])) if len(arr) >= p else float(arr[-1])

    def sigmoid(x, k=10):
        return 1.0 / (1.0 + np.exp(-x * k))

    def clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    ma20, ma50 = ma(closes, 20), ma(closes, 50)
    ma200 = ma(closes, 200) if n >= 200 else ma50

    scores = []

    # T1: 价格 vs MA20 偏离
    dev20 = (current - ma20) / ma20 if ma20 > 0 else 0
    scores.append(clip01(sigmoid(dev20, 50)))

    # T2: 价格 vs MA50
    dev50 = (current - ma50) / ma50 if ma50 > 0 else 0
    scores.append(clip01(sigmoid(dev50, 30)))

    # T3: 价格 vs MA200
    dev200 = (current - ma200) / ma200 if ma200 > 0 else 0
    scores.append(clip01(sigmoid(dev200, 20)))

    # T4: MA排列
    if ma20 > ma50 > ma200:
        scores.append(1.0)
    elif ma20 < ma50 < ma200:
        scores.append(0.0)
    else:
        scores.append(0.5)

    # T5-T7: 布林带
    bb_std = float(np.std(closes[-20:])) if n >= 20 else 0
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_range = bb_upper - bb_lower
    if bb_range > 0:
        scores.append(clip01((current - bb_lower) / bb_range))       # T5: 布林位置
        scores.append(clip01((bb_upper - current) / bb_range))       # T6: 距上轨
        scores.append(clip01(bb_range / ma20))                       # T7: 带宽
    else:
        scores.extend([0.5, 0.5, 0.5])

    # T8: RSI14
    if n >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_g, avg_l = np.mean(gains), np.mean(losses)
        rsi = float(100 - 100 / (1 + avg_g / avg_l)) if avg_l > 0 else 100.0
    else:
        rsi = 50.0
    scores.append(clip01(rsi / 100.0))

    # T9: ATR%
    trs = []
    for i in range(1, min(15, n)):
        tr = max(highs[-i] - lows[-i],
                 abs(highs[-i] - closes[-i - 1]),
                 abs(lows[-i] - closes[-i - 1]))
        trs.append(tr)
    atr = float(np.mean(trs)) if trs else 0
    atr_pct = atr / current if current > 0 else 0
    scores.append(clip01(atr_pct * 20))

    # T10-T13: 动量
    for period in [1, 5, 10, 20]:
        if n > period:
            momentum = (closes[-1] - closes[-1 - period]) / closes[-1 - period]
            scores.append(clip01(sigmoid(momentum, 15 + period)))
        else:
            scores.append(0.5)

    # T14-T17: 量比
    for period in [1, 5, 10, 20]:
        if n > period:
            vol_curr = float(np.mean(volumes[-period:])) if period > 1 else float(volumes[-1])
            vol_prev = float(np.mean(volumes[-2*period:-period]))
            ratio = vol_curr / vol_prev if vol_prev > 0 else 1.0
            # 量比 → [0,1], 1.0=正常, >2=放量, <0.5=缩量
            scores.append(clip01((ratio - 0.5) / 2.0))
        else:
            scores.append(0.5)

    # T18-T21: 距高低位
    for period in [20, 50]:
        p = min(period, n)
        hh = float(np.max(highs[-p:]))
        ll = float(np.min(lows[-p:]))
        rng = hh - ll
        if rng > 0:
            scores.append(clip01((current - ll) / rng))   # 距低位
            scores.append(clip01((hh - current) / rng))    # 距高位
        else:
            scores.extend([0.5, 0.5])

    # T22-T24: 波动率分位
    for period in [20, 50, 200]:
        p = min(period, n)
        if p >= 5:
            rets = np.diff(closes[-p:]) / closes[-p:-1]
            vol = float(np.std(rets))
            # 当前波动率 vs 历史
            if n >= p * 2:
                hist_rets = np.diff(closes[-2*p:-p]) / closes[-2*p:-p-1]
                hist_vol = float(np.std(hist_rets))
            else:
                hist_vol = vol
            ratio = vol / hist_vol if hist_vol > 0 else 1.0
            scores.append(clip01(ratio / 2.0))
        else:
            scores.append(0.5)

    return scores


# ============================================================
# 兼容接口 (对应 binance_data.py 的函数签名)
# ============================================================
def get_klines_for_technical(name: str, count: int = 200) -> List[Dict]:
    with GtradeDataProvider(use_ws=False) as p:
        return p.get_klines(name, interval="1d", limit=count)


# ============================================================
# 自检
# ============================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    print("=" * 60)
    print("GtradeDataProvider — gTrade TradFi 永续自检")
    print(f"代理: {PROXY_URL or '无 (直连)'}")
    print("=" * 60)

    # 品种列表
    pairs = get_tradfi_pairs()
    print(f"\n📊 TradFi 品种: {len(pairs)} 个")

    for cat in ['crypto', 'stocks', 'indices', 'commodities']:
        items = [p for p in pairs if p['group'] == cat]
        print(f"\n  {cat} ({len(items)}个):")
        for p in items[:6]:
            print(f"    [{p['pairIndex']:>4}] {p['symbol']:<16} spread={p['spread_pct']}%")

    # 测试 Yahoo Finance K线
    print(f"\n📈 NVDA K线测试 (yfinance):")
    with GtradeDataProvider(use_ws=False) as p:
        klines = p.get_klines("NVDA", limit=10)
        if klines:
            latest = klines[-1]
            print(f"  最新: ${latest['close']:.2f}  ({datetime.fromtimestamp(latest['open_time']/1000).strftime('%Y-%m-%d')})")
            print(f"  数据条数: {len(klines)}")

        tech = p.get_technical_indicators("NVDA")
        if tech['price']:
            print(f"  MA20=${tech['ma20']:.2f}  MA50=${tech['ma50']:.2f}  RSI={tech['rsi14']:.0f}")
            print(f"  布林: [${tech['bb_lower']:.2f}, ${tech['bb_upper']:.2f}]")
            print(f"  均线排列: {tech['ma_alignment']}")

        # 市场状态
        status = p.get_market_status()
        print(f"\n🏛️ 市场状态: stocks={'🟢开' if status['stocks'] else '🔴关'}")

    # 测试 WebSocket
    print(f"\n🔌 WebSocket v4 测试:")
    with GtradeDataProvider(use_ws=True) as p:
        time.sleep(1)
        nvda_pidx = get_pair_index_by_name("NVDA")
        aapl_pidx = get_pair_index_by_name("AAPL")
        if nvda_pidx:
            p_nvda = p.get_price(nvda_pidx)
            print(f"  NVDA mark: ${p_nvda}" if p_nvda else "  NVDA: 无数据")
        if aapl_pidx:
            p_aapl = p.get_price(aapl_pidx)
            print(f"  AAPL mark: ${p_aapl}" if p_aapl else "  AAPL: 无数据")

        total_prices = len(p._prices)
        print(f"  WebSocket 覆盖: {total_prices} 品种")

    print(f"\n🎉 自检完成!")
