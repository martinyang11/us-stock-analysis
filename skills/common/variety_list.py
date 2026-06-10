#!/usr/bin/env python3
"""
统一币种列表 — 唯一权威数据源
供 CANN、CatTrader 和其他模块导入使用
"""

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

CATEGORY_MAP = {
    **{i: "L1" for i in range(16)},
    **{i: "L2" for i in range(16, 23)},
    **{i: "DeFi" for i in range(23, 34)},
    **{i: "Meme" for i in range(34, 41)},
    **{i: "AI" for i in range(41, 45)},
    **{i: "Infra" for i in range(45, 50)},
}


def get_variety_info(vid: int):
    """返回 (名称, 代码, 类别)"""
    name = VARIETY_NAMES.get(vid, f"品种{vid}")
    code = VARIETY_CODES.get(vid, "")
    cat = CATEGORY_MAP.get(vid, "Other")
    return name, code, cat


def get_variety_category_by_id(vid: int) -> str:
    return CATEGORY_MAP.get(vid, "Other")
