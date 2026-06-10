#!/usr/bin/env python3
"""
统一TradFi品种列表 — 唯一权威数据源
供 CANN、CatTrader 和其他模块导入
Binance USDT-M TradFi永续合约（美股个股+ETF+商品）
"""

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

CATEGORY_MAP = {
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


def get_variety_info(vid: int):
    """返回 (名称, 代码, 类别)"""
    name = VARIETY_NAMES.get(vid, f"品种{vid}")
    code = VARIETY_CODES.get(vid, "")
    cat = CATEGORY_MAP.get(vid, "Other")
    return name, code, cat


def get_variety_category_by_id(vid: int) -> str:
    return CATEGORY_MAP.get(vid, "Other")
