#!/usr/bin/env python3
"""
统一 TradFi 品种列表 — gTrade 版
供 SANN、CatTrader 和其他模块导入

品种来源: gTrade /trading-variables/all → 过滤 stocks/indices/commodities/crypto
"""

import os
import sys
import json

# 延迟加载，避免循环导入
_variety_loaded = False

# gTrade pairIndex → 品种元数据
GTRADE_PAIR_MAP: dict = {}  # {pairIndex: {name, group, spread_pct, ...}}
# variety_id (0..N) → gTrade pairIndex
VID_TO_PAIR_INDEX: dict = {}  # {vid: pairIndex}
# gTrade pairIndex → variety_id
PAIR_INDEX_TO_VID: dict = {}  # {pairIndex: vid}
# variety_id → 名称
VARIETY_NAMES: dict = {}  # {vid: "NVDA"}
# variety_id → 代码
VARIETY_CODES: dict = {}  # {vid: "NVDA"}
# 名称 → variety_id
SYMBOL_TO_ID: dict = {}  # {"NVDA": vid}
# 品种总数
NUM_VARIETIES = 0
# variety_id → 类别
CATEGORY_MAP: dict = {}


def _load():
    """延迟加载品种列表（首次调用时从 gtrade_data 加载）"""
    global _variety_loaded, NUM_VARIETIES
    if _variety_loaded:
        return

    # 导入 gtrade_data
    sa_dir = os.path.join(os.path.dirname(__file__), '..', 'StockAnalysis', 'scripts')
    if sa_dir not in sys.path:
        sys.path.insert(0, sa_dir)

    from gtrade_data import get_variety_list as gvl

    varieties = gvl()
    if not varieties:
        return

    NUM_VARIETIES = len(varieties)
    for info in varieties:
        vid = info['variety_id']
        pair_idx = info['pairIndex']
        name = info['name']

        GTRADE_PAIR_MAP[pair_idx] = info
        VID_TO_PAIR_INDEX[vid] = pair_idx
        PAIR_INDEX_TO_VID[pair_idx] = vid
        VARIETY_NAMES[vid] = name
        VARIETY_CODES[vid] = name
        SYMBOL_TO_ID[name] = vid
        CATEGORY_MAP[vid] = info.get('category', 'Other')

    _variety_loaded = True


def get_variety_info(vid: int):
    """返回 (名称, 代码, 类别)"""
    _load()
    name = VARIETY_NAMES.get(vid, f"品种{vid}")
    code = VARIETY_CODES.get(vid, "")
    cat = CATEGORY_MAP.get(vid, "Other")
    return name, code, cat


def get_variety_category_by_id(vid: int) -> str:
    """返回品种类别"""
    _load()
    return CATEGORY_MAP.get(vid, "Other")


def get_pair_index(vid: int) -> int:
    """variety_id → gTrade pairIndex"""
    _load()
    return VID_TO_PAIR_INDEX.get(vid, -1)


def get_vid_from_pair_index(pair_idx: int) -> int:
    """gTrade pairIndex → variety_id"""
    _load()
    return PAIR_INDEX_TO_VID.get(pair_idx, -1)


# 直接加载（首次导入时）
_load()

# SYMBOLS = 品种名列表（兼容旧代码）
SYMBOLS = list(VARIETY_NAMES.values()) if _variety_loaded else []
