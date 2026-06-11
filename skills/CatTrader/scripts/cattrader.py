#!/usr/bin/env python3
"""
CatTrader — 基于SANN评分的TradFi趋势跟踪交易系统 v3.4

核心逻辑：
  每4小时执行一次决策循环：
  1. 读取最新SA评分（14基本面+24技术面，由SANN管线统一计算）
  2. 加载SANN模型推理，生成35品种综合评分s
  3. 查表得到每个品种的目标杠杆（方向+倍数）
  4. 选取目标杠杆最强的1多1空作为操作品种
  5. 与当前持仓对比，派生开仓/调仓/平仓/持有动作

仓位映射表（s → 目标杠杆）：
  s ≤ 0.35       → 做空 0.5×（σ≥0.15）
  0.35 < s ≤ 0.45 → 做空 0.3×（σ∈[0.05,0.15)）
  0.45 < s < 0.55 → 平仓（σ<0.05 中性区）
  0.55 ≤ s < 0.65 → 做多 0.3×（σ∈[0.05,0.15)）
  s ≥ 0.65       → 做多 0.5×（σ≥0.15）

TradFi版保护：
  - 资金费率 > 0.1% → 做多仓位减半
  - 资金费率 < -0.1% → 做空仓位减半
  - OI达到30日极值 → 标记警告
"""

import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np

# ===== 环境自检 =====
for pkg in ['numpy']:
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

# 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SA_SCRIPTS = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'StockAnalysis', 'scripts'))
SANN_SCRIPTS = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'SANN', 'scripts'))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'common'))
for p in [PROJECT_ROOT, SA_SCRIPTS, SANN_SCRIPTS, COMMON_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from skills.common.variety_list import (
    VARIETY_NAMES, VARIETY_CODES, SYMBOLS, NUM_VARIETIES
)
from binance_data import BinanceDataProvider

logger = logging.getLogger('CatTrader')


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
    direction: str
    leverage: float
    zone: str


@dataclass
class Position:
    crypto_id: int
    crypto_name: str
    symbol: str
    direction: str
    entry_score: float
    entry_sigma: float
    leverage: float
    entry_time: str
    entry_date: str


@dataclass
class Decision:
    action: str
    crypto_id: int
    crypto_name: str
    symbol: str
    direction: str
    leverage: float
    score: float
    sigma: float
    zone: str
    reason: str
    funding_warning: str = ""


@dataclass
class TraderState:
    positions: List[dict] = field(default_factory=list)
    last_run: str = ""
    run_count: int = 0
    history: List[dict] = field(default_factory=list)


# ============================================================
# 仓位映射表
# ============================================================
def get_target_leverage(score: float) -> TargetLeverage:
    """SANN评分 → 目标杠杆"""
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
    return abs(score - 0.5)


# ============================================================
# 资金费率保护（加密版新增）
# ============================================================
def apply_funding_protection(target: TargetLeverage, funding_rate: float) -> TargetLeverage:
    """资金费率极端时降仓保护

    - 资金费率 > 0.1% → 做多拥挤，多头仓位减半
    - 资金费率 < -0.1% → 做空拥挤，空头仓位减半
    """
    if target.leverage == 0:
        return target

    if funding_rate > 0.001:  # >0.1%
        if target.direction == Direction.LONG.value:
            new_lev = target.leverage / 2
            return TargetLeverage(target.direction, new_lev,
                                 f"{target.zone}(费率保护)")

    if funding_rate < -0.001:  # <-0.1%
        if target.direction == Direction.SHORT.value:
            new_lev = target.leverage / 2
            return TargetLeverage(target.direction, new_lev,
                                 f"{target.zone}(费率保护)")

    return target


# ============================================================
# 币种筛选
# ============================================================
def select_candidates(scores: Dict[int, float]) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
    """选取cmax和cmin"""
    long_candidates = {}
    short_candidates = {}
    for cid, s in scores.items():
        target = get_target_leverage(s)
        if target.direction == Direction.LONG.value:
            long_candidates[cid] = s
        elif target.direction == Direction.SHORT.value:
            short_candidates[cid] = s

    cmax = max(long_candidates.items(), key=lambda x: x[1]) if long_candidates else None
    cmin = min(short_candidates.items(), key=lambda x: x[1]) if short_candidates else None
    return cmax, cmin


# ============================================================
# SANN评分获取
# ============================================================
def _find_latest_ca_scores(scores_dir: str, date_str: str) -> Tuple[Dict[int, List[float]], str]:
    """找到最新含CA评分的CSV"""
    import csv as csv_mod
    from datetime import timedelta

    base_date = datetime.strptime(date_str, '%Y%m%d')
    for offset in range(31):
        check_date = (base_date - timedelta(days=offset)).strftime('%Y%m%d')
        csv_path = os.path.join(scores_dir, f'scores_{check_date}.csv')
        if not os.path.exists(csv_path):
            continue

        ca_scores = {}
        has_any = False
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                # 兼容 crypto_id 和 variety_id
                cid = int(row.get('crypto_id', row.get('variety_id', -1)))
                if cid < 0:
                    continue
                dim1 = float(row.get('dim1', '-1'))
                if dim1 >= 0:
                    dims_14 = [float(row[f'dim{i}']) for i in range(1, 15)]
                    techs_24 = [float(row[f'tech{i}']) for i in range(1, 25)]
                    # 合并为38维（14+24）
                    dims = dims_14 + techs_24
                    if min(dims_14) >= 0:
                        ca_scores[cid] = dims
                        has_any = True

        if has_any:
            trading_days = offset * 5 // 7
            if trading_days > 15:
                logger.critical(f"CA评分已超过15日未更新({check_date})，建议暂停交易")
            elif trading_days > 5:
                logger.warning(f"CA评分已超过5日未更新({check_date})")
            return ca_scores, check_date

    return {}, ''


_model_cache = None
_model_cache_time = 0.0


def _load_model_cached(data_dir: str):
    """加载SANN模型（带缓存）"""
    global _model_cache, _model_cache_time
    now = time.time()
    if _model_cache is not None and (now - _model_cache_time) < 300:
        return _model_cache

    if SANN_SCRIPTS not in sys.path:
        sys.path.insert(0, SANN_SCRIPTS)

    try:
        from pretrain_numpy import load_pretrained_model
        model, path = load_pretrained_model(data_dir)
        if model is not None:
            _model_cache = model
            _model_cache_time = now
            return model
    except ImportError:
        pass
    return None


def get_cann_scores(date_str: str = None) -> Tuple[Dict[int, float], str, Dict[int, float]]:
    """获取50币种SANN评分 + 资金费率

    Returns:
        (scores_dict, ca_date, funding_rates_dict)
    """
    if date_str is None:
        date_str = datetime.utcnow().strftime('%Y%m%d')

    cann_data_dir = os.path.join(PROJECT_ROOT, 'skills', 'SANN', 'data')
    scores_dir = os.path.join(cann_data_dir, 'daily_scores')
    month = int(date_str[4:6])

    # CA评分
    ca_scores, ca_date = _find_latest_ca_scores(scores_dir, date_str)

    # SANN模型
    model = _load_model_cached(cann_data_dir)
    if model is None:
        logger.warning("无SANN模型，全部返回0.5")
        return {cid: 0.5 for cid in range(NUM_VARIETIES)}, ca_date, {}

    # 导入推理函数
    try:
        from pretrain_numpy import predict_single
    except ImportError:
        return {cid: 0.5 for cid in range(NUM_VARIETIES)}, ca_date, {}

    # 全币种推理
    result = {}
    for cid in range(NUM_VARIETIES):
        if cid not in ca_scores:
            result[cid] = 0.5
            continue
        score = predict_single(model, ca_scores[cid], month, cid)
        result[cid] = score

    # 获取资金费率
    funding_rates = {}
    try:
        with BinanceDataProvider() as provider:
            funding_rates = provider.get_all_funding_rates_dict()
    except Exception:
        pass

    return result, ca_date, funding_rates


# ============================================================
# 状态管理
# ============================================================
STATE_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'data', 'state.json')


def load_state() -> TraderState:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            positions = []
            for p in data.get('positions', []):
                if 'position_pct' in p and 'leverage' not in p:
                    p['leverage'] = p.pop('position_pct')
                if 'variety_id' in p and 'crypto_id' not in p:
                    p['crypto_id'] = p.pop('variety_id')
                if 'variety_name' in p and 'crypto_name' not in p:
                    p['crypto_name'] = p.pop('variety_name')
                positions.append(p)
            return TraderState(
                positions=positions,
                last_run=data.get('last_run', ''),
                run_count=data.get('run_count', 0),
                history=data.get('history', [])[-100:],
            )
        except Exception:
            pass
    return TraderState()


def save_state(state: TraderState):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(asdict(state), f, ensure_ascii=False, indent=2)


# ============================================================
# 决策循环
# ============================================================
def _resolve_position(current: Optional[Position], target: TargetLeverage,
                      score: float, cid: int, now: datetime, date_str: str,
                      funding_rate: float = 0.0) -> List[Decision]:
    """根据当前持仓和目标杠杆派生决策"""
    name = VARIETY_NAMES.get(cid, f"币种{cid}")
    symbol = VARIETY_CODES.get(cid, "")
    sigma = compute_sigma(score)
    decisions = []

    # 资金费率警告
    funding_warning = ""
    if funding_rate > 0.001:
        funding_warning = f"⚠️费率{funding_rate*100:.3f}%极高，做多拥挤"
    elif funding_rate < -0.001:
        funding_warning = f"⚠️费率{funding_rate*100:.3f}%极低，做空拥挤"

    if current is None:
        if target.leverage > 0:
            decisions.append(Decision(
                action=Action.OPEN.value, crypto_id=cid, crypto_name=name,
                symbol=symbol, direction=target.direction, leverage=target.leverage,
                score=score, sigma=sigma, zone=target.zone,
                reason=f"{target.zone}区间 s={score:.4f} σ={sigma:.4f}",
                funding_warning=funding_warning,
            ))
        return decisions

    if target.leverage == 0:
        decisions.append(Decision(
            action=Action.CLOSE.value, crypto_id=cid, crypto_name=name,
            symbol=symbol, direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"s={score:.4f}进入平仓区，信号消失",
        ))
    elif target.direction != current.direction:
        decisions.append(Decision(
            action=Action.CLOSE.value, crypto_id=cid, crypto_name=name,
            symbol=symbol, direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号反转{current.direction}→{target.direction}",
        ))
        decisions.append(Decision(
            action=Action.OPEN.value, crypto_id=cid, crypto_name=name,
            symbol=symbol, direction=target.direction, leverage=target.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号反转开{target.direction}{target.leverage}× s={score:.4f}",
            funding_warning=funding_warning,
        ))
    elif abs(target.leverage - current.leverage) > 0.01:
        decisions.append(Decision(
            action=Action.ADJUST.value, crypto_id=cid, crypto_name=name,
            symbol=symbol, direction=current.direction, leverage=target.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"{current.leverage}×→{target.leverage}× {target.zone}",
        ))
    else:
        decisions.append(Decision(
            action=Action.HOLD.value, crypto_id=cid, crypto_name=name,
            symbol=symbol, direction=current.direction, leverage=current.leverage,
            score=score, sigma=sigma, zone=target.zone,
            reason=f"信号维持 σ={sigma:.4f}",
        ))

    return decisions


def run_cat_trader(date_str: str = None) -> dict:
    """执行CatTrader决策循环"""
    now = datetime.utcnow()
    if date_str is None:
        date_str = now.strftime('%Y%m%d')

    state = load_state()
    positions = [Position(**p) for p in state.positions]

    # Step 1: 获取SANN评分+资金费率
    scores, ca_date, funding_rates = get_cann_scores(date_str)

    # Step 2: 选取候选
    cmax, cmin = select_candidates(scores)

    # Step 3: 处理现有持仓
    decisions = []
    positions_after = []
    processed_ids = set()

    for pos in positions:
        current_score = scores.get(pos.crypto_id, 0.5)
        target = get_target_leverage(current_score)
        fr = funding_rates.get(pos.crypto_id, 0.0)
        target = apply_funding_protection(target, fr)
        pos_decisions = _resolve_position(pos, target, current_score, pos.crypto_id, now, date_str, fr)
        decisions.extend(pos_decisions)

        closed = any(d.action == Action.CLOSE.value for d in pos_decisions)
        reversed_open = any(d.action == Action.OPEN.value and d.direction != pos.direction for d in pos_decisions)

        if not closed:
            adjusted = any(d.action == Action.ADJUST.value for d in pos_decisions)
            if adjusted:
                pos.leverage = target.leverage
            positions_after.append(pos)
        elif reversed_open:
            positions_after.append(Position(
                crypto_id=pos.crypto_id, crypto_name=pos.crypto_name,
                symbol=pos.symbol, direction=target.direction,
                entry_score=current_score, entry_sigma=compute_sigma(current_score),
                leverage=target.leverage, entry_time=now.isoformat(),
                entry_date=date_str,
            ))
        processed_ids.add(pos.crypto_id)

    # Step 4: 处理候选
    for candidate, fallback_dir in [(cmax, Direction.LONG.value), (cmin, Direction.SHORT.value)]:
        if candidate is None:
            decisions.append(Decision(
                action=Action.SKIP.value, crypto_id=-1, crypto_name="无",
                symbol="", direction=fallback_dir, leverage=0.0,
                score=0.5, sigma=0.0, zone="无",
                reason=f"无{fallback_dir}候选（无币种s越过中性区）",
            ))
            continue

        cid, score = candidate
        if cid in processed_ids:
            continue

        target = get_target_leverage(score)
        fr = funding_rates.get(cid, 0.0)
        target = apply_funding_protection(target, fr)
        pos_decisions = _resolve_position(None, target, score, cid, now, date_str, fr)
        decisions.extend(pos_decisions)

        opened = any(d.action == Action.OPEN.value for d in pos_decisions)
        if opened:
            positions_after.append(Position(
                crypto_id=cid,
                crypto_name=VARIETY_NAMES.get(cid, f"币种{cid}"),
                symbol=VARIETY_CODES.get(cid, ""),
                direction=target.direction,
                entry_score=score,
                entry_sigma=compute_sigma(score),
                leverage=target.leverage,
                entry_time=now.isoformat(),
                entry_date=date_str,
            ))
        processed_ids.add(cid)

    # 更新状态
    state.positions = [asdict(p) for p in positions_after]
    state.last_run = now.isoformat()
    state.run_count += 1

    for d in decisions:
        if d.action in (Action.OPEN.value, Action.CLOSE.value, Action.ADJUST.value):
            state.history.append({'time': now.isoformat(), **asdict(d)})
    state.history = state.history[-100:]
    save_state(state)

    return generate_report(decisions, scores, positions_after, date_str,
                          state, ca_date, funding_rates)


# ============================================================
# 报告生成
# ============================================================
def generate_report(decisions: List[Decision], scores: Dict[int, float],
                    positions: List[Position], date_str: str, state: TraderState,
                    ca_date: str = '', funding_rates: Dict[int, float] = None) -> dict:
    now = datetime.utcnow()
    fr = funding_rates or {}
    score_values = list(scores.values()) if scores else [0.5]
    score_arr = np.array(score_values)

    opens = [d for d in decisions if d.action == Action.OPEN.value]
    closes = [d for d in decisions if d.action == Action.CLOSE.value]
    holds = [d for d in decisions if d.action == Action.HOLD.value]
    adjusts = [d for d in decisions if d.action == Action.ADJUST.value]

    zone_counts = {"空0.5×": 0, "空0.3×": 0, "平仓": 0, "多0.3×": 0, "多0.5×": 0}
    for s in score_values:
        t = get_target_leverage(s)
        zone_counts[t.zone] += 1

    sorted_by_sigma = sorted(
        [(cid, compute_sigma(s)) for cid, s in scores.items()],
        key=lambda x: x[1], reverse=True
    )
    top5 = [(VARIETY_NAMES.get(cid, "?"), cid, scores[cid], sigma,
             fr.get(cid, 0))
            for cid, sigma in sorted_by_sigma[:5]]

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
            'skips': len([d for d in decisions if d.action == Action.SKIP.value]),
            'details': [asdict(d) for d in decisions],
        },
        'positions': {
            'total': len(positions),
            'long': len([p for p in positions if p.direction == Direction.LONG.value]),
            'short': len([p for p in positions if p.direction == Direction.SHORT.value]),
            'details': [asdict(p) for p in positions],
        },
        'funding_warnings': [
            {'crypto': name, 'rate': f'{rate*100:.4f}%'}
            for name, cid, s, sig, rate in top5 if abs(rate) > 0.0005
        ],
        'top5_sigma': [
            {'name': name, 'crypto_id': cid, 'score': round(s, 4),
             'sigma': round(sig, 4), 'funding_rate': f'{rate*100:.4f}%'}
            for name, cid, s, sig, rate in top5
        ],
    }


def format_report_text(report: dict) -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("CatTrader 决策报告 (Crypto)")
    lines.append(f"时间: {report['run_time'][:19]} UTC  第{report['run_count']}次运行")
    if report.get('ca_date'):
        lines.append(f"CA数据: {report['ca_date']}")
    lines.append("=" * 55)

    ss = report['score_stats']
    lines.append(f"\n📊 SANN评分统计")
    lines.append(f"  均值={ss['mean']:.4f} σ={ss['std']:.4f}  范围=[{ss['min']:.4f}, {ss['max']:.4f}]")

    zd = report['zone_distribution']
    lines.append(f"\n📈 区间分布(50币种)")
    lines.append(f"  空0.5×:{zd['空0.5×']}  空0.3×:{zd['空0.3×']}  平仓:{zd['平仓']}  多0.3×:{zd['多0.3×']}  多0.5×:{zd['多0.5×']}")

    dd = report['decisions']
    lines.append(f"\n📋 决策摘要")
    lines.append(f"  开仓:{dd['opens']}  平仓:{dd['closes']}  调仓:{dd['adjusts']}  持有:{dd['holds']}  空缺:{dd['skips']}")

    for d in dd['details']:
        sym = d.get('symbol', '')
        fw = f" {d.get('funding_warning', '')}" if d.get('funding_warning') else ""
        if d['action'] == '空缺':
            lines.append(f"  ⚪ {d['direction']}: {d['reason']}")
        elif d['action'] == '开仓':
            lines.append(f"  🟢 开仓 {d['crypto_name']}({sym}) {d['direction']} {d['leverage']}× | {d['zone']} s={d['score']:.4f}{fw}")
        elif d['action'] == '平仓':
            lines.append(f"  🔴 平仓 {d['crypto_name']}({sym}) {d['direction']} | s={d['score']:.4f}")
        elif d['action'] == '调仓':
            lines.append(f"  🔄 调仓 {d['crypto_name']}({sym}) {d['direction']} {d['leverage']}× | {d['reason']}")
        elif d['action'] == '持有':
            lines.append(f"  🔵 持有 {d['crypto_name']}({sym}) {d['direction']} {d['leverage']}× | σ={d['sigma']:.4f}")

    pp = report['positions']
    lines.append(f"\n💼 当前持仓")
    lines.append(f"  多头:{pp['long']}  空头:{pp['short']}")
    for p in pp['details']:
        emoji = "📈" if p['direction'] == '多头' else "📉"
        lines.append(f"  {emoji} {p['crypto_name']}({p.get('symbol','')}) {p['direction']} {p['leverage']}× | 入场s={p['entry_score']:.4f}")

    if report.get('funding_warnings'):
        lines.append(f"\n⚠️ 资金费率警告")
        for w in report['funding_warnings']:
            lines.append(f"  {w['crypto']}: {w['rate']}")

    lines.append(f"\n🏆 信号最强TOP5")
    for t in report['top5_sigma']:
        lines.append(f"  {t['name']}: s={t['score']:.4f} σ={t['sigma']:.4f} 费率={t['funding_rate']}")

    lines.append("\n" + "=" * 55)
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='CatTrader (Crypto)')
    parser.add_argument('--date', default=None, help='日期 YYYYMMDD')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    report = run_cat_trader(args.date)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report_text(report))
