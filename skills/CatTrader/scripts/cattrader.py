#!/usr/bin/env python3
"""
CatTrader — 基于SANN评分的TradFi趋势跟踪交易系统 v3.4

核心逻辑：
  每4小时执行一次决策循环：
  1. 读取最新SA评分（14基本面+24技术面，由SANN管线统一计算）
  2. 加载SANN模型推理，生成56品种综合评分s
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
  - gTrade spread > 0.6% → gTrade高spread→仓位降一档
  - gTrade spread > 0.6% → gTrade高spread→仓位降一档
  - isStocksOpen=false → 不新开仓 (市场关闭保护)
"""

import os
import sys
import subprocess
import json
import time
import logging
from datetime import datetime, timezone
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
from gtrade_data import GtradeDataProvider

logger = logging.getLogger('CatTrader')


# ============================================================
# 常量
# ============================================================
SL_PCT = 0.15  # 15% 止损
TP_PCT = 0.15  # 15% 止盈


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
    entry_price: float = 0.0  # 入场价格 (用于SL/TP计算)


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
    spread_warning: str = ""


@dataclass
class TraderState:
    positions: List[dict] = field(default_factory=list)
    last_run: str = ""
    run_count: int = 0
    history: List[dict] = field(default_factory=list)


# ============================================================
# 链上交易执行
# ============================================================
def _get_onchain_adapter():
    """延迟加载链上交易适配器（避免影响纯分析模式）"""
    import sys, os
    onchain_dir = os.path.join(SCRIPT_DIR, '..', '..', '..', 'onchain_trade')
    if onchain_dir not in sys.path:
        sys.path.insert(0, onchain_dir)
    from hole_board.exchange.onchain.account_config import parse_onchain_config
    from hole_board.exchange.onchain.venues.gains.adapter import GainsVenueAdapter
    from dt_config.onchain_config import WALLET_ADDRESS, PRIVATE_KEY, ONCHAIN_DEFAULTS

    config = parse_onchain_config(
        api_name='okx_onchain',
        user_id=WALLET_ADDRESS,
        password=PRIVATE_KEY,
        onchain_venue=ONCHAIN_DEFAULTS.get('onchain_venue', 'gains'),
        chain_id=ONCHAIN_DEFAULTS.get('chain_id', 42161),
        rpc_url=ONCHAIN_DEFAULTS.get('rpc_url', 'https://arb1.arbitrum.io/rpc'),
        dry_run=ONCHAIN_DEFAULTS.get('dry_run', True),
    )
    return GainsVenueAdapter(config), ONCHAIN_DEFAULTS.get('dry_run', True)


def execute_onchain(decisions: List[Decision], positions: List[Position],
                    state: TraderState, date_str: str) -> List[str]:
    """将 CatTrader 决策执行到链上 gTrade。

    只在有 OPEN/CLOSE 决策时调用。
    返回执行日志列表。
    """
    onchain_logs = []
    try:
        adapter, is_dry = _get_onchain_adapter()
    except Exception as e:
        logger.warning(f'链上适配器加载失败（可能未配置钱包）: {e}')
        return onchain_logs

    tag = '[DRY-RUN]' if is_dry else '[ONCHAIN]'
    logger.info(f'{tag} 开始执行链上交易 (dry_run={is_dry})')

    # 市场状态检查 + 价格源 (gTrade WebSocket)
    from gtrade_data import GtradeDataProvider
    try:
        gtp = GtradeDataProvider(use_ws=True)  # 启用 WebSocket 获取实时价格
        market_status = gtp.get_market_status()
        stocks_open = market_status.get('stocks', False)
    except Exception:
        gtp = None
        stocks_open = False

    # 将 WebSocket 价格源注入链上适配器（优先于 Chainlink / yfinance）
    try:
        adapter.set_price_provider(gtp)
    except Exception:
        pass

    for d in decisions:
        symbol = d.symbol or VARIETY_CODES.get(d.crypto_id, '')
        if not symbol:
            onchain_logs.append(f'跳过 {d.crypto_name}: 无 symbol 映射')
            continue

        # 检查市场状态
        cat = VARIETY_NAMES.get(d.crypto_id, '')
        is_crypto = d.crypto_id in (0, 1)  # BTC, ETH
        if not is_crypto and not stocks_open:
            onchain_logs.append(f'跳过 {d.crypto_name}({symbol}): 市场关闭')
            continue

        try:
            if d.action == Action.OPEN.value:
                # 开仓: collateral 固定 5 USDC, leverage 2x (gTrade最低)
                collateral = 5.0
                leverage = 2.0  # gTrade 最低杠杆 2x

                from hole_board.exchange.onchain.types import OnchainOpenRequest
                req = OnchainOpenRequest(
                    symbol=symbol, side='long' if d.direction == Direction.LONG.value else 'short',
                    collateral=collateral, leverage=leverage, slippage=0.01,
                )
                result = adapter.open_trade(req)
                msg = f'{tag} 开仓 {symbol} {d.direction} {leverage:.1f}x {collateral}USDC → tx={result.tx_hash}'
                logger.info(msg)
                onchain_logs.append(msg)

                # 记录链上 trade_index 到状态
                if result.order_sys_id:
                    for p in state.positions:
                        if p.get('crypto_id') == d.crypto_id:
                            p['onchain_trade_index'] = result.order_sys_id

            elif d.action == Action.CLOSE.value:
                # 平仓: 找到对应的链上仓位
                from hole_board.exchange.onchain.types import OnchainCloseRequest

                # 先查链上持仓
                onchain_positions = adapter.fetch_positions() or []
                closed = False

                # 按 symbol 匹配平仓
                for op in onchain_positions:
                    from hole_board.exchange.onchain.venues.gains.adapter import _PAIR_SYMBOLS
                    op_symbol = _PAIR_SYMBOLS.get(op.get('pair_index', -1), '')
                    if op_symbol.upper() == symbol.upper():
                        req = OnchainCloseRequest(
                            symbol=symbol,
                            position_id=str(op['index']),
                            slippage=0.01,
                        )
                        result = adapter.close_trade(req)
                        msg = f'{tag} 平仓 {symbol} #{op["index"]} → tx={result.tx_hash}'
                        logger.info(msg)
                        onchain_logs.append(msg)
                        closed = True
                        break

                if not closed:
                    onchain_logs.append(f'{tag} 平仓 {symbol} 失败: 链上无匹配持仓')

        except Exception as e:
            msg = f'{tag} 执行失败 {symbol} {d.action}: {e}'
            logger.warning(msg)
            onchain_logs.append(msg)

    return onchain_logs


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
# 止损止盈检查
# ============================================================
def check_sl_tp(position: Position, current_price: float) -> Optional[str]:
    """检查是否触发止损或止盈

    Returns:
        'stop_loss' | 'take_profit' | None
    """
    if position.entry_price <= 0 or current_price <= 0:
        return None

    if position.direction == Direction.LONG.value:
        sl_price = position.entry_price * (1 - SL_PCT)
        tp_price = position.entry_price * (1 + TP_PCT)
        if current_price <= sl_price:
            return 'stop_loss'
        if current_price >= tp_price:
            return 'take_profit'
    else:  # SHORT
        sl_price = position.entry_price * (1 + SL_PCT)
        tp_price = position.entry_price * (1 - TP_PCT)
        if current_price >= sl_price:
            return 'stop_loss'
        if current_price <= tp_price:
            return 'take_profit'

    return None


def _fetch_current_prices(vids: List[int]) -> Dict[int, float]:
    """获取指定品种的当前价格"""
    prices = {}
    try:
        from skills.StockAnalysis.scripts.gtrade_data import get_pair_index_by_name
        with GtradeDataProvider(use_ws=False) as p:
            for vid in vids:
                name = VARIETY_NAMES.get(vid, '')
                # 按品种名从yfinance获取最新价格
                klines = p.get_klines(name, limit=2)
                if klines:
                    prices[vid] = klines[-1]['close']
    except Exception:
        pass
    return prices


# ============================================================
# Spread保护 (gTrade版)
# ============================================================
def apply_spread_protection(target: TargetLeverage, spread: float) -> TargetLeverage:
    """资金费率极端时降仓保护

    - gTrade spread > 0.6% → 做多拥挤，多头仓位减半
    - gTrade spread > 0.6% → 做空拥挤，空头仓位减半
    """
    if target.leverage == 0:
        return target

    if spread > 0.001:  # >0.1%
        if target.direction == Direction.LONG.value:
            new_lev = target.leverage / 2
            return TargetLeverage(target.direction, new_lev,
                                 f"{target.zone}(费率保护)")

    if spread < -0.001:  # <-0.1%
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


def get_sann_scores(date_str: str = None) -> Tuple[Dict[int, float], str, Dict[int, float]]:
    """获取50币种SANN评分 + 资金费率

    Returns:
        (scores_dict, ca_date, spreads_dict)
    """
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')

    sann_data_dir = os.path.join(PROJECT_ROOT, 'skills', 'SANN', 'data')
    scores_dir = os.path.join(sann_data_dir, 'daily_scores')
    month = int(date_str[4:6])

    # CA评分
    ca_scores, ca_date = _find_latest_ca_scores(scores_dir, date_str)

    # SANN模型
    model = _load_model_cached(sann_data_dir)
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
    spreads = {}
    try:
        with GtradeDataProvider() as provider:
            spreads = provider.get_all_spreads_dict()
    except Exception:
        pass

    return result, ca_date, spreads


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
                      spread: float = 0.0) -> List[Decision]:
    """根据当前持仓和目标杠杆派生决策"""
    name = VARIETY_NAMES.get(cid, f"币种{cid}")
    symbol = VARIETY_CODES.get(cid, "")
    sigma = compute_sigma(score)
    decisions = []

    # 资金费率警告
    spread_warning = ""
    if spread > 0.001:
        spread_warning = f"⚠️费率{spread*100:.3f}%极高，做多拥挤"
    elif spread < -0.001:
        spread_warning = f"⚠️费率{spread*100:.3f}%极低，做空拥挤"

    if current is None:
        if target.leverage > 0:
            decisions.append(Decision(
                action=Action.OPEN.value, crypto_id=cid, crypto_name=name,
                symbol=symbol, direction=target.direction, leverage=target.leverage,
                score=score, sigma=sigma, zone=target.zone,
                reason=f"{target.zone}区间 s={score:.4f} σ={sigma:.4f}",
                spread_warning=spread_warning,
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
            spread_warning=spread_warning,
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


def ensure_daily_data(date_str: str):
    """确保今日 SA 评分 + SANN 推理结果存在，不存在则自动生成。

    在 CatTrader 决策前调用，实现 SA → SANN → CatTrader 全自动管线。
    """
    import subprocess

    scores_dir = os.path.join(PROJECT_ROOT, 'skills', 'SANN', 'data', 'daily_scores')
    scores_csv = os.path.join(scores_dir, f'scores_{date_str}.csv')
    cann_json = os.path.join(scores_dir, f'cann_results_{date_str}.json')

    # 1. SA 评分
    sa_script = os.path.join(PROJECT_ROOT, 'scripts', 'run_sa_scoring.py')
    if not os.path.exists(scores_csv):
        logger.info(f'[AUTO] SA 评分不存在，自动生成: {scores_csv}')
        try:
            subprocess.run(
                [sys.executable, sa_script],
                cwd=PROJECT_ROOT, check=True, timeout=600,
                capture_output=True, text=True,
                env={**os.environ, 'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:7897')},
            )
            logger.info('[AUTO] SA 评分生成完成')
        except subprocess.TimeoutExpired:
            logger.error('[AUTO] SA 评分超时（10分钟），跳过')
        except Exception as e:
            logger.error(f'[AUTO] SA 评分失败: {e}')

    # 2. SANN 推理（仅推理，不训练）
    if os.path.exists(scores_csv) and not os.path.exists(cann_json):
        logger.info(f'[AUTO] SANN 推理结果不存在，自动生成: {cann_json}')
        try:
            sann_script = os.path.join(PROJECT_ROOT, 'skills', 'SANN', 'scripts', 'daily_pipeline.py')
            subprocess.run(
                [sys.executable, sann_script, '--date', date_str,
                 '--data-dir', os.path.join(PROJECT_ROOT, 'skills', 'SANN', 'data'),
                 '--inference-only'],
                cwd=PROJECT_ROOT, check=True, timeout=300,
                capture_output=True, text=True,
                env={**os.environ, 'HTTPS_PROXY': os.environ.get('HTTPS_PROXY', 'http://127.0.0.1:7897')},
            )
            logger.info('[AUTO] SANN 推理生成完成')
        except subprocess.TimeoutExpired:
            logger.error('[AUTO] SANN 推理超时（5分钟），跳过')
        except Exception as e:
            logger.error(f'[AUTO] SANN 推理失败: {e}')


def run_cat_trader(date_str: str = None) -> dict:
    """执行CatTrader决策循环"""
    now = datetime.now(timezone.utc)
    if date_str is None:
        date_str = now.strftime('%Y%m%d')

    # Step 0: 自动准备 SA + SANN 数据
    ensure_daily_data(date_str)

    state = load_state()
    positions = [Position(**p) for p in state.positions]

    # Step 1: 获取SANN评分+资金费率
    scores, ca_date, spreads = get_sann_scores(date_str)

    # Step 2: 选取候选
    cmax, cmin = select_candidates(scores)

    # Step 3: 获取当前价格 + 检查止损止盈
    decisions = []
    positions_after = []
    processed_ids = set()

    # 获取所有持仓品种的当前价格 (用于SL/TP检查)
    held_vids = [p.crypto_id for p in positions if p.entry_price <= 0 or True]
    current_prices = _fetch_current_prices(held_vids) if held_vids else {}

    # 先检查已有持仓的SL/TP
    for pos in positions[:]:  # 用切片避免迭代时修改
        current_price = current_prices.get(pos.crypto_id, 0)
        sl_tp = check_sl_tp(pos, current_price)
        if sl_tp:
            name = VARIETY_NAMES.get(pos.crypto_id, '?')
            emoji = '🛑' if sl_tp == 'stop_loss' else '🎯'
            pct = SL_PCT if sl_tp == 'stop_loss' else TP_PCT
            reason = f'{emoji} {sl_tp}: {pos.direction}入场@{pos.entry_price:.2f} → 现价@{current_price:.2f} (触发{int(pct*100)}%)'
            decisions.append(Decision(
                action=Action.CLOSE.value, crypto_id=pos.crypto_id, crypto_name=name,
                symbol=pos.symbol, direction=pos.direction, leverage=pos.leverage,
                score=scores.get(pos.crypto_id, 0.5), sigma=0, zone=sl_tp,
                reason=reason,
            ))
            positions.remove(pos)
            processed_ids.add(pos.crypto_id)
            continue

    for pos in positions:
        current_score = scores.get(pos.crypto_id, 0.5)
        target = get_target_leverage(current_score)
        fr = spreads.get(pos.crypto_id, 0.0)
        target = apply_spread_protection(target, fr)
        pos_decisions = _resolve_position(pos, target, current_score, pos.crypto_id, now, date_str, fr)
        decisions.extend(pos_decisions)

        closed = any(d.action == Action.CLOSE.value for d in pos_decisions)
        reversed_open = any(d.action == Action.OPEN.value and d.direction != pos.direction for d in pos_decisions)

        if not closed:
            adjusted = any(d.action == Action.ADJUST.value for d in pos_decisions)
            if adjusted:
                pos.leverage = target.leverage
            # 每次运行时用最新市价更新入场价，SL/TP随之浮动
            cur_px = current_prices.get(pos.crypto_id, 0)
            if cur_px > 0:
                pos.entry_price = cur_px
            positions_after.append(pos)
        elif reversed_open:
            ep = current_prices.get(pos.crypto_id, pos.entry_price)
            positions_after.append(Position(
                crypto_id=pos.crypto_id, crypto_name=pos.crypto_name,
                symbol=pos.symbol, direction=target.direction,
                entry_score=current_score, entry_sigma=compute_sigma(current_score),
                leverage=target.leverage, entry_time=now.isoformat(),
                entry_date=date_str, entry_price=ep,
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
        fr = spreads.get(cid, 0.0)
        target = apply_spread_protection(target, fr)
        pos_decisions = _resolve_position(None, target, score, cid, now, date_str, fr)
        decisions.extend(pos_decisions)

        opened = any(d.action == Action.OPEN.value for d in pos_decisions)
        if opened:
            # 获取入场价格
            entry_price = current_prices.get(cid, 0)
            if entry_price <= 0:
                # fallback: 单独获取
                ep_dict = _fetch_current_prices([cid])
                entry_price = ep_dict.get(cid, 0)
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
                entry_price=entry_price,
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

    # 执行链上交易（OPEN/CLOSE决策）
    report = generate_report(decisions, scores, positions_after, date_str,
                             state, ca_date, spreads)
    onchain_logs = execute_onchain(decisions, positions_after, state, date_str)
    if onchain_logs:
        report['onchain_logs'] = onchain_logs

    return report


# ============================================================
# 报告生成
# ============================================================
def generate_report(decisions: List[Decision], scores: Dict[int, float],
                    positions: List[Position], date_str: str, state: TraderState,
                    ca_date: str = '', spreads: Dict[int, float] = None) -> dict:
    now = datetime.now(timezone.utc)
    fr = spreads or {}
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
        'spread_warnings': [
            {'crypto': name, 'rate': f'{rate*100:.4f}%'}
            for name, cid, s, sig, rate in top5 if abs(rate) > 0.0005
        ],
        'top5_sigma': [
            {'name': name, 'crypto_id': cid, 'score': round(s, 4),
             'sigma': round(sig, 4), 'spread': f'{rate*100:.4f}%'}
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
        fw = f" {d.get('spread_warning', '')}" if d.get('spread_warning') else ""
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
        ep = p.get('entry_price', 0)
        if ep > 0:
            sl = ep * (1 - SL_PCT) if p['direction'] == '多头' else ep * (1 + SL_PCT)
            tp = ep * (1 + TP_PCT) if p['direction'] == '多头' else ep * (1 - TP_PCT)
            sl_tp_info = f" | 入场价=${ep:.2f} SL=${sl:.2f} TP=${tp:.2f}"
        else:
            sl_tp_info = ""
        lines.append(f"  {emoji} {p['crypto_name']}({p.get('symbol','')}) {p['direction']} {p['leverage']}× | 入场s={p['entry_score']:.4f}{sl_tp_info}")

    if report.get('spread_warnings'):
        lines.append(f"\n⚠️ 资金费率警告")
        for w in report['spread_warnings']:
            lines.append(f"  {w['crypto']}: {w['rate']}")

    lines.append(f"\n🏆 信号最强TOP5")
    for t in report['top5_sigma']:
        lines.append(f"  {t['name']}: s={t['score']:.4f} σ={t['sigma']:.4f} 费率={t['spread']}")

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
