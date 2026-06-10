#!/usr/bin/env python3
"""
CatTrader — 基于CANN评分的趋势跟踪交易系统 v3.2

核心逻辑：
  每个交易时段执行一次决策循环：
  1. 读取最新38维CA评分（14基本面+24技术面，由CANN 16:00管线统一计算）
  2. 加载CANN模型推理，38维CA评分直接输入，生成55品种综合评分s
  3. 查表得到每个品种的目标杠杆（方向+倍数）
  4. 选取目标杠杆最强的1多1空作为操作品种
  5. 与当前持仓对比，派生开仓/调仓/平仓/持有动作

仓位映射表（s → 目标杠杆，唯一权威定义）：
  s ≤ 0.35       → 做空 0.5倍杠杆（σ≥0.15 特别显著）
  0.35 < s ≤ 0.45 → 做空 0.3倍杠杆（σ∈[0.05,0.15) 一般显著）
  0.45 < s < 0.55 → 平仓 0倍杠杆（σ<0.05 中性区）
  0.55 ≤ s < 0.65 → 做多 0.3倍杠杆（σ∈[0.05,0.15) 一般显著）
  s ≥ 0.65       → 做多 0.5倍杠杆（σ≥0.15 特别显著）

开仓约束（铁律）：
  品种必须处于平仓区之外（s≤0.45或s≥0.55）才能开仓。
  中性区（0.45<s<0.55）的品种永远不会被选为候选，
  也永远不会触发开仓动作。此约束由select_candidates()
  排除中性区品种+get_target_leverage()映射为Direction.NONE
  双重保证，无需额外判断。

杠杆含义：
  每个品种独立拥有杠杆倍数作为仓位水平。
  例：铜0.5× 表示铜品种使用0.5倍杠杆做多，
      焦煤0.3× 表示焦煤品种使用0.3倍杠杆做空。
  多品种杠杆独立，不叠加计算。

决策推导（统一规则）：
  当前=无仓, 目标≠0 → 开仓（仅平仓区外品种）
  当前=有仓, 目标=0 → 平仓（信号消失）
  当前=有仓, 同向同杠杆 → 持有
  当前=有仓, 同向不同杠杆 → 调仓
  当前=有仓, 反向 → 平仓+开仓

数据流：
  CANN 16:00管线负责每日微调模型+CA全维度评分+写入样本
  CatTrader 每次运行：读最新38维CA评分 + 推理 → 决策（v4.0：技术面已归入CA引擎）

状态持久化: skills/CatTrader/data/state.json
"""

import os
import sys
import subprocess

# ===== 环境自检：确保依赖包可用 =====
for pkg in ['numpy', 'tqsdk']:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[自检] 安装缺失依赖: {pkg}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
        print(f"[自检] {pkg} 安装完成")

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'common'))
for p in [PROJECT_ROOT, COMMON_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger('CatTrader')

from skills.common.variety_list import VARIETY_NAMES, VARIETY_CODES, NUM_VARIETIES


# ============================================================
# 常量
# ============================================================

class Direction(Enum):
    LONG = "多头"
    SHORT = "空头"
    NONE = "无"


class Action(Enum):
    OPEN = "开仓"
    CLOSE = "平仓"
    ADJUST = "调仓"
    HOLD = "持有"
    SKIP = "空缺"


# ============================================================
# 数据结构
# ============================================================

@dataclass
class TargetLeverage:
    """目标杠杆（查表结果）"""
    direction: str       # "多头" / "空头" / "无"
    leverage: float      # 0.0 / 0.3 / 0.5
    zone: str            # "多0.5×" / "多0.3×" / "平仓" / "空0.3×" / "空0.5×"


@dataclass
class Position:
    """持仓记录"""
    variety_id: int
    variety_name: str
    direction: str           # "多头" / "空头"
    entry_score: float       # 开仓时CANN评分
    entry_sigma: float       # 开仓时信号强度
    leverage: float          # 杠杆倍数 0.3 / 0.5
    entry_time: str          # 开仓时间 ISO格式
    entry_date: str          # 开仓日期 YYYYMMDD
    contract_code: str = ""  # 主力合约代码（如ag2608）


@dataclass
class Decision:
    """交易决策"""
    action: str              # "开仓"/"平仓"/"调仓"/"持有"/"空缺"
    variety_id: int
    variety_name: str
    direction: str           # "多头"/"空头"/"无"
    leverage: float          # 杠杆倍数
    score: float             # 当前CANN评分
    sigma: float             # 信号强度
    zone: str                # 所属区间
    reason: str              # 决策理由
    contract_code: str = ""  # 主力合约代码


@dataclass
class TraderState:
    """CatTrader持久化状态"""
    positions: List[dict] = field(default_factory=list)
    last_run: str = ""
    run_count: int = 0
    history: List[dict] = field(default_factory=list)


# ============================================================
# 合约代码解析
# ============================================================

_contract_cache: dict = {}  # {code: contract_code}


def _resolve_contract_codes(codes: List[str]) -> Dict[str, str]:
    """批量解析品种代码→主力合约代码（如cu→cu2608），带缓存"""
    global _contract_cache
    unresolved = [c for c in codes if c not in _contract_cache or not _contract_cache.get(c)]  # 空字符串=未解析, 允许重试
    if not unresolved:
        return {c: _contract_cache[c] for c in codes}

    try:
        from tqsdk import TqApi, TqAuth
        import os
        auth = TqAuth(
            os.environ.get('TQSDK_USER', 'ardio2001'),
            os.environ.get('TQSDK_PASS', 'ardio801104'),
        )
        api = TqApi(auth=auth)
        for code in unresolved:
            # 映射品种代码到交易所格式
            exchange_map = {
                'cu': 'SHFE', 'al': 'SHFE', 'zn': 'SHFE', 'pb': 'SHFE', 'ni': 'SHFE', 'sn': 'SHFE',
                'au': 'SHFE', 'ag': 'SHFE', 'rb': 'SHFE', 'hc': 'SHFE', 'ss': 'SHFE',
                'bu': 'SHFE', 'ru': 'SHFE', 'sp': 'SHFE', 'wr': 'SHFE',
                'fu': 'SHFE', 'lu': 'INE', 'sc': 'INE', 'bc': 'INE', 'nr': 'INE',
                'a': 'DCE', 'b': 'DCE', 'c': 'DCE', 'cs': 'DCE', 'eb': 'DCE', 'eg': 'DCE',
                'i': 'DCE', 'j': 'DCE', 'jm': 'DCE', 'l': 'DCE', 'm': 'DCE', 'p': 'DCE',
                'pg': 'DCE', 'pp': 'DCE', 'rr': 'DCE', 'v': 'DCE', 'y': 'DCE',
                'CF': 'CZCE', 'CY': 'CZCE', 'FG': 'CZCE', 'MA': 'CZCE', 'OI': 'CZCE',
                'PF': 'CZCE', 'PK': 'CZCE', 'RM': 'CZCE', 'SA': 'CZCE', 'SF': 'CZCE',
                'SH': 'CZCE', 'SM': 'CZCE', 'SR': 'CZCE', 'TA': 'CZCE', 'UR': 'CZCE',
                'ZC': 'CZCE', 'AP': 'CZCE', 'CJ': 'CZCE', 'JR': 'CZCE', 'LR': 'CZCE',
                'PM': 'CZCE', 'RI': 'CZCE', 'RS': 'CZCE', 'WH': 'CZCE',
                'lc': 'GFEX', 'si': 'GFEX',
            }
            exchange = exchange_map.get(code, 'SHFE')
            try:
                quote = api.get_quote(f"KQ.m@{exchange}.{code}")
                api.wait_update()
                ul = quote.get('underlying_symbol', '')
                _contract_cache[code] = ul
                logger.debug(f"合约解析: {code} → {ul}")
            except Exception:
                _contract_cache[code] = ''
        api.close()
    except Exception as e:
        logger.warning(f"合约解析失败: {e}，使用空合约代码")
        for code in unresolved:
            _contract_cache[code] = ''

    return {c: _contract_cache.get(c, '') for c in codes}


# ============================================================
# 核心逻辑：仓位映射表
# ============================================================

def get_target_leverage(score: float) -> TargetLeverage:
    """根据CANN评分s查表得到目标杠杆

    s ≤ 0.35       → 空0.5×  特别显著
    0.35 < s ≤ 0.45 → 空0.3×  一般显著
    0.45 < s < 0.55 → 平仓   中性区
    0.55 ≤ s < 0.65 → 多0.3×  一般显著
    s ≥ 0.65       → 多0.5×  特别显著
    """
    if score <= 0.35:
        return TargetLeverage(Direction.SHORT.value, 0.5, "空0.5×")
    elif score <= 0.45:
        return TargetLeverage(Direction.SHORT.value, 0.3, "空0.3×")
    elif score < 0.55:
        return TargetLeverage(Direction.NONE.value, 0.0, "平仓")
    elif score < 0.65:
        return TargetLeverage(Direction.LONG.value, 0.3, "多0.3×")
    else:
        return TargetLeverage(Direction.LONG.value, 0.5, "多0.5×")


def compute_sigma(score: float) -> float:
    """计算信号强度 σ = |s - 0.5|"""
    return abs(score - 0.5)


def select_candidates(scores: Dict[int, float]) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
    """从55品种中选取cmax和cmin

    cmax: 目标为做多的品种中s最大的（σ最大的多头品种）
    cmin: 目标为做空的品种中s最小的（σ最大的空头品种）

    ⚠️ 开仓约束（铁律）：品种必须处于平仓区之外才能开仓。
    此函数只从target.direction为LONG或SHORT的品种中筛选，
    中性区(0.45<s<0.55, Direction.NONE)的品种自动排除，
    确保cmax/cmin永远不会指向中性区品种。
    """
    long_candidates = {}
    short_candidates = {}
    for vid, s in scores.items():
        target = get_target_leverage(s)
        if target.direction == Direction.LONG.value:
            long_candidates[vid] = s
        elif target.direction == Direction.SHORT.value:
            short_candidates[vid] = s

    cmax = max(long_candidates.items(), key=lambda x: x[1]) if long_candidates else None
    cmin = min(short_candidates.items(), key=lambda x: x[1]) if short_candidates else None

    return cmax, cmin


# ============================================================
# CANN评分获取：读取最新38维CA评分 + 推理（v4.0：技术面已归入CA引擎）
# ============================================================

# VARIETY_CODES 已从 skills.common.variety_list 统一导入（唯一权威源）


def _find_latest_ca_scores(scores_dir: str, date_str: str) -> Tuple[Dict[int, List[float]], str]:
    """从daily_scores目录找到最新的含CA评分的CSV，返回{vid: [dim1..dim14]}

    从date_str往前最多搜索30天，找到第一个dim1>=0（已填充CA）的CSV。
    超过5个交易日未更新输出WARNING，超过15个交易日输出CRITICAL。
    返回: (ca_scores_dict, actual_date_str)  — ca_scores_dict values为38维列表
    """
    import csv as csv_mod
    from datetime import timedelta

    base_date = datetime.strptime(date_str, '%Y%m%d')
    for offset in range(31):
        check_date = (base_date - timedelta(days=offset)).strftime('%Y%m%d')
        csv_path = os.path.join(scores_dir, f'scores_{check_date}.csv')
        if not os.path.exists(csv_path):
            continue

        ca_scores = {}
        has_any_filled = False
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            fieldnames = reader.fieldnames or []
            # 兼容旧格式(dim1-14+tech1-24)和新格式(dim1-38)
            is_old_format = 'tech1' in fieldnames and 'dim15' not in fieldnames
            for row in reader:
                vid = int(row['variety_id'])
                dim1 = float(row.get('dim1', '-1'))
                if dim1 >= 0:
                    if is_old_format:
                        dims_14 = [float(row[f'dim{i}']) for i in range(1, 15)]
                        techs_24 = [float(row[f'tech{i}']) for i in range(1, 25)]
                        dims = dims_14 + techs_24
                    else:
                        dims = [float(row[f'dim{i}']) for i in range(1, 39)]
                    if min(dims) >= 0:
                        ca_scores[vid] = dims
                        has_any_filled = True

        if has_any_filled:
            # 时效性告警
            trading_days = offset * 5 // 7  # 粗估交易日
            if trading_days > 15:
                logger.critical(f"CA评分已超过15个交易日未更新({check_date})，建议暂停交易")
            elif trading_days > 5:
                logger.warning(f"CA评分已超过5个交易日未更新({check_date})，数据可能过时")
            return ca_scores, check_date

    return {}, ''


# CANN模型缓存（避免每次运行重新加载）
_model_cache = None
_model_cache_time = 0.0
_MODEL_CACHE_TTL = 300  # 5分钟缓存


def _load_model_cached(data_dir: str):
    """加载CANN模型（带5分钟缓存，避免每次运行重新加载）"""
    global _model_cache, _model_cache_time
    now = time.time()
    if _model_cache is not None and (now - _model_cache_time) < _MODEL_CACHE_TTL:
        return _model_cache

    # 确保CANN脚本目录在path中
    cann_scripts_dir = os.path.join(PROJECT_ROOT, 'skills', 'CANN', 'scripts')
    if cann_scripts_dir not in sys.path:
        sys.path.insert(0, cann_scripts_dir)

    try:
        from pretrain_numpy import load_pretrained_model
        model, path = load_pretrained_model(data_dir)
        if model is not None:
            _model_cache = model
            _model_cache_time = now
            logger.info(f"CANN模型已加载: {path}")
            return model
    except ImportError:
        pass
    return None


def get_cann_scores(date_str: str = None) -> Tuple[Dict[int, float], str]:
    """获取55品种CANN评分 — 读取最新38维CA评分 + 推理（v4.0：技术面已归入CA）

    数据流：
    1. 从daily_scores目录找到最新含CA评分的CSV，读取dim1-dim38（38维CA统一评分）
    2. 加载CANN模型（16:00管线微调后的权重），全品种推理
    3. 无CA数据的品种s=0.5（冷启动保护）

    模型由16:00每日管线微调，CatTrader只做推理。
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')

    cann_data_dir = os.path.join(PROJECT_ROOT, 'skills', 'CANN', 'data')
    scores_dir = os.path.join(cann_data_dir, 'daily_scores')
    month = int(date_str[4:6])

    # Step 1: 读取最新38维CA评分
    ca_scores, ca_date = _find_latest_ca_scores(scores_dir, date_str)
    if ca_date:
        logger.info(f"CA评分来源: scores_{ca_date}.csv（{len(ca_scores)}品种有数据）")
    else:
        logger.warning("未找到任何含CA评分的CSV，全部s=0.5")

    # Step 2: 加载CANN模型（带缓存）
    model = _load_model_cached(cann_data_dir)
    if model is None:
        logger.warning("无CANN模型权重，全部返回0.5")
        return {vid: 0.5 for vid in range(NUM_VARIETIES)}, ca_date

    # Step 3: 全品种推理（38维CA评分直接输入，无需单独采集技术面）
    result = {}
    for vid in range(NUM_VARIETIES):
        if vid not in ca_scores:
            result[vid] = 0.5
            continue
        score = _predict(model, ca_scores[vid], month, vid)
        result[vid] = score

    return result, ca_date



def _predict(model, dim_scores, month, variety_id):
    """推理封装（延迟导入predict_single）— v4.0: 38维CA评分直接输入"""
    cann_scripts_dir = os.path.join(PROJECT_ROOT, 'skills', 'CANN', 'scripts')
    if cann_scripts_dir not in sys.path:
        sys.path.insert(0, cann_scripts_dir)

    try:
        from pretrain_numpy import predict_single
    except ImportError:
        return 0.5

    return predict_single(model, dim_scores, month, variety_id)


# ============================================================
# 状态管理
# ============================================================

STATE_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'state.json')


def load_state() -> TraderState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容旧版position_pct字段
            positions = []
            for p in data.get('positions', []):
                if 'position_pct' in p and 'leverage' not in p:
                    p['leverage'] = p.pop('position_pct')
                positions.append(p)
            return TraderState(
                positions=positions,
                last_run=data.get('last_run', ''),
                run_count=data.get('run_count', 0),
                history=data.get('history', [])[-100:],
            )
        except Exception as e:
            logger.warning(f"状态文件加载失败: {e}，使用空状态")
    return TraderState()


def save_state(state: TraderState):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)


def positions_from_dicts(pos_dicts: List[dict]) -> List[Position]:
    """从dict列表重建Position列表（load_state已处理position_pct→leverage兼容）"""
    return [Position(**p) for p in pos_dicts]


# ============================================================
# 主决策循环
# ============================================================

def _resolve_position(current: Optional[Position], target: TargetLeverage,
                      score: float, vid: int, now: datetime, date_str: str) -> List[Decision]:
    """根据当前持仓和目标杠杆，派生决策列表

    统一规则：
      当前=无仓, 目标≠0 → 开仓
      当前=有仓, 目标=0 → 平仓
      当前=有仓, 同向同杠杆 → 持有
      当前=有仓, 同向不同杠杆 → 调仓
      当前=有仓, 反向 → 平仓+开仓
    """
    name = VARIETY_NAMES.get(vid, f"品种{vid}")
    sigma = compute_sigma(score)
    decisions = []

    # 无当前持仓
    if current is None:
        if target.leverage > 0:
            decisions.append(Decision(
                action=Action.OPEN.value, variety_id=vid, variety_name=name,
                direction=target.direction, leverage=target.leverage,
                score=score, sigma=sigma, zone=target.zone,
                reason=f"{target.zone}区间，s={score:.4f} σ={sigma:.4f}",
            ))
        return decisions

    # 有当前持仓
    if target.leverage == 0:
        # 目标=平仓
        decisions.append(Decision(
            action=Action.CLOSE.value, variety_id=vid, variety_name=name,
            direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"s={score:.4f}进入{target.zone}区，信号消失",
        ))
    elif target.direction != current.direction:
        # 反向 → 平仓+开仓
        decisions.append(Decision(
            action=Action.CLOSE.value, variety_id=vid, variety_name=name,
            direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号反转{current.direction}→{target.direction}，先平仓",
        ))
        decisions.append(Decision(
            action=Action.OPEN.value, variety_id=vid, variety_name=name,
            direction=target.direction, leverage=target.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号反转，开{target.direction}{target.leverage}×，s={score:.4f}",
        ))
    elif abs(target.leverage - current.leverage) > 0.01:
        # 同向不同杠杆 → 调仓
        decisions.append(Decision(
            action=Action.ADJUST.value, variety_id=vid, variety_name=name,
            direction=current.direction, leverage=target.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"{current.leverage}×→{target.leverage}×，{target.zone}区间",
        ))
    else:
        # 同向同杠杆 → 持有
        decisions.append(Decision(
            action=Action.HOLD.value, variety_id=vid, variety_name=name,
            direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号维持{target.zone}区，σ={sigma:.4f}",
        ))

    return decisions


def run_cat_trader(date_str: str = None) -> dict:
    """执行CatTrader决策循环"""
    now = datetime.now()
    if date_str is None:
        date_str = now.strftime('%Y%m%d')

    state = load_state()
    positions = positions_from_dicts(state.positions)

    # Step 1: 获取CANN评分
    scores, ca_date = get_cann_scores(date_str)

    # Step 2: 选取候选品种
    cmax, cmin = select_candidates(scores)

    # 解析主力合约代码
    needed_codes = set()
    for pos in positions:
        needed_codes.add(VARIETY_CODES.get(pos.variety_id, ''))
    if cmax:
        needed_codes.add(VARIETY_CODES.get(cmax[0], ''))
    if cmin:
        needed_codes.add(VARIETY_CODES.get(cmin[0], ''))
    needed_codes.discard('')
    contract_map = _resolve_contract_codes(list(needed_codes)) if needed_codes else {}
    # 补全存量持仓中缺失的合约代码
    for pos in positions:
        if not pos.contract_code:
            code = VARIETY_CODES.get(pos.variety_id, '')
            pos.contract_code = contract_map.get(code, '')

    # Step 3: 处理现有持仓（查表→对比→派生决策）
    decisions = []
    positions_after = []
    processed_vids = set()

    for pos in positions:
        current_score = scores.get(pos.variety_id, 0.5)
        target = get_target_leverage(current_score)
        pos_decisions = _resolve_position(pos, target, current_score, pos.variety_id, now, date_str)
        decisions.extend(pos_decisions)

        # 判断持仓是否保留
        closed = any(d.action == Action.CLOSE.value for d in pos_decisions)
        reversed_open = any(d.action == Action.OPEN.value and d.direction != pos.direction for d in pos_decisions)

        if not closed:
            adjusted = any(d.action == Action.ADJUST.value for d in pos_decisions)
            if adjusted:
                pos.leverage = target.leverage
            positions_after.append(pos)
        elif reversed_open:
            new_pos = Position(
                variety_id=pos.variety_id,
                variety_name=pos.variety_name,
                direction=target.direction,
                entry_score=current_score,
                entry_sigma=compute_sigma(current_score),
                leverage=target.leverage,
                entry_time=now.isoformat(),
                entry_date=date_str,
                contract_code=pos.contract_code,
            )
            positions_after.append(new_pos)

        processed_vids.add(pos.variety_id)

    # Step 4: 处理候选品种（cmax/cmin）
    for candidate, fallback_direction in [(cmax, Direction.LONG.value), (cmin, Direction.SHORT.value)]:
        if candidate is None:
            decisions.append(Decision(
                action=Action.SKIP.value, variety_id=-1, variety_name="无",
                direction=fallback_direction, leverage=0.0,
                score=0.5, sigma=0.0, zone="无",
                reason=f"无{fallback_direction}候选（无品种s越过中性区）",
            ))
            continue

        vid, score = candidate
        if vid in processed_vids:
            continue

        target = get_target_leverage(score)
        pos_decisions = _resolve_position(None, target, score, vid, now, date_str)
        decisions.extend(pos_decisions)

        opened = any(d.action == Action.OPEN.value for d in pos_decisions)
        if opened:
            positions_after.append(Position(
                variety_id=vid,
                variety_name=VARIETY_NAMES.get(vid, f"品种{vid}"),
                direction=target.direction,
                entry_score=score,
                entry_sigma=compute_sigma(score),
                leverage=target.leverage,
                entry_time=now.isoformat(),
                entry_date=date_str,
                contract_code=contract_map.get(VARIETY_CODES.get(vid, ''), ''),
            ))
        processed_vids.add(vid)

    # 更新状态
    state.positions = [asdict(p) for p in positions_after]
    state.last_run = now.isoformat()
    state.run_count += 1

    # 批量注入决策的合约代码（从已解析的持仓/候选映射中获取）
    for d in decisions:
        if d.variety_id >= 0 and not d.contract_code:
            code = VARIETY_CODES.get(d.variety_id, '')
            d.contract_code = contract_map.get(code, '')

    for d in decisions:
        if d.action in (Action.OPEN.value, Action.CLOSE.value, Action.ADJUST.value):
            state.history.append({'time': now.isoformat(), **asdict(d)})
    state.history = state.history[-100:]

    save_state(state)

    return generate_report(decisions, scores, positions_after, date_str, state, ca_date)


# ============================================================
# 报告生成
# ============================================================

def generate_report(decisions: List[Decision], scores: Dict[int, float],
                    positions: List[Position], date_str: str, state: TraderState,
                    ca_date: str = '') -> dict:
    now = datetime.now()

    score_values = list(scores.values()) if scores else [0.5]
    score_arr = np.array(score_values)

    opens = [d for d in decisions if d.action == Action.OPEN.value]
    closes = [d for d in decisions if d.action == Action.CLOSE.value]
    holds = [d for d in decisions if d.action == Action.HOLD.value]
    skips = [d for d in decisions if d.action == Action.SKIP.value]
    adjusts = [d for d in decisions if d.action == Action.ADJUST.value]

    # 区间分布统计
    zone_counts = {"空0.5×": 0, "空0.3×": 0, "平仓": 0, "多0.3×": 0, "多0.5×": 0}
    for s in score_values:
        t = get_target_leverage(s)
        zone_counts[t.zone] += 1

    sorted_by_sigma = sorted(
        [(vid, compute_sigma(s)) for vid, s in scores.items()],
        key=lambda x: x[1], reverse=True
    )
    top5 = [(VARIETY_NAMES.get(vid, f"?"), vid, scores[vid], sigma)
            for vid, sigma in sorted_by_sigma[:5]]

    return {
        'run_time': now.isoformat(),
        'date': date_str,
        'ca_date': ca_date,
        'run_count': state.run_count,
        'score_stats': {
            'mean': float(np.mean(score_arr)),
            'std': float(np.std(score_arr)),
            'min': float(np.min(score_arr)),
            'max': float(np.max(score_arr)),
        },
        'zone_distribution': zone_counts,
        'decisions': {
            'opens': len(opens),
            'closes': len(closes),
            'adjusts': len(adjusts),
            'holds': len(holds),
            'skips': len(skips),
            'details': [asdict(d) for d in decisions],
        },
        'positions': {
            'total': len(positions),
            'long': len([p for p in positions if p.direction == Direction.LONG.value]),
            'short': len([p for p in positions if p.direction == Direction.SHORT.value]),
            'details': [asdict(p) for p in positions],
        },
        'top5_sigma': [
            {'name': name, 'variety_id': vid, 'score': round(s, 4), 'sigma': round(sig, 4)}
            for name, vid, s, sig in top5
        ],
    }


def format_report_text(report: dict) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("CatTrader 决策报告")
    lines.append(f"时间: {report['run_time'][:19]}  第{report['run_count']}次运行")
    if report.get('ca_date'):
        lines.append(f"CA数据: {report['ca_date']}")
    lines.append("=" * 55)

    ss = report['score_stats']
    lines.append(f"\n📊 CANN评分统计")
    lines.append(f"  均值={ss['mean']:.4f}  σ={ss['std']:.4f}")
    lines.append(f"  范围=[{ss['min']:.4f}, {ss['max']:.4f}]")

    zd = report['zone_distribution']
    lines.append(f"\n📈 区间分布")
    lines.append(f"  空0.5×:{zd['空0.5×']}  空0.3×:{zd['空0.3×']}  平仓:{zd['平仓']}  多0.3×:{zd['多0.3×']}  多0.5×:{zd['多0.5×']}")

    dd = report['decisions']
    lines.append(f"\n📋 决策摘要")
    lines.append(f"  开仓:{dd['opens']}  平仓:{dd['closes']}  调仓:{dd['adjusts']}  持有:{dd['holds']}  空缺:{dd['skips']}")

    for d in dd['details']:
        cc = f" {d.get('contract_code', '')}" if d.get('contract_code') else ""
        if d['action'] == '空缺':
            lines.append(f"  ⚪ {d['direction']}: {d['reason']}")
        elif d['action'] == '开仓':
            lines.append(f"  🟢 开仓 {d['variety_name']}{cc} {d['direction']} {d['leverage']}× | {d['zone']} s={d['score']:.4f} σ={d['sigma']:.4f}")
        elif d['action'] == '平仓':
            lines.append(f"  🔴 平仓 {d['variety_name']}{cc} {d['direction']} | s={d['score']:.4f} | {d['reason']}")
        elif d['action'] == '调仓':
            lines.append(f"  🔄 调仓 {d['variety_name']}{cc} {d['direction']} {d['leverage']}× | {d['reason']}")
        elif d['action'] == '持有':
            lines.append(f"  🔵 持有 {d['variety_name']}{cc} {d['direction']} {d['leverage']}× | {d['zone']} σ={d['sigma']:.4f}")

    pp = report['positions']
    lines.append(f"\n💼 当前持仓")
    lines.append(f"  多头:{pp['long']}  空头:{pp['short']}")
    for p in pp['details']:
        emoji = "📈" if p['direction'] == '多头' else "📉"
        cc = f" {p.get('contract_code', '')}" if p.get('contract_code') else ""
        lines.append(f"  {emoji} {p['variety_name']}{cc} {p['direction']} {p['leverage']}× | 入场s={p['entry_score']:.4f}")

    lines.append(f"\n🏆 信号最强TOP5")
    for t in report['top5_sigma']:
        zone = get_target_leverage(t['score']).zone
        lines.append(f"  {t['name']}: s={t['score']:.4f} σ={t['sigma']:.4f} [{zone}]")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


# ============================================================
# CLI入口
# ============================================================

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='CatTrader 交易决策系统')
    parser.add_argument('--date', default=None, help='日期 YYYYMMDD，默认今天')
    parser.add_argument('--json', action='store_true', help='仅输出JSON')
    args = parser.parse_args()

    report = run_cat_trader(args.date)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report_text(report))
