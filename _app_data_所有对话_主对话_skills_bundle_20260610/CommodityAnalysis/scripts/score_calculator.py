#!/usr/bin/env python3
"""
商品期货基本面分析 - 评分计算器
根据14维度评分和权重，计算加权综合得分
"""

import json


def calculate_weighted_score(scores: dict, weights: dict) -> dict:
    """
    计算加权综合得分

    Args:
        scores: 各维度得分, e.g. {"1_货币政策": 0.6, "2_地缘政治": 0.5, ...}
        weights: 各维度权重, e.g. {"1_货币政策": 0.10, "2_地缘政治": 0.08, ...}

    Returns:
        完整评估结果
    """
    # P1-9: 校验评分范围 [0,1]
    for dim, score in scores.items():
        if not isinstance(score, (int, float)):
            raise ValueError(f"评分必须是数值类型，维度 {dim} 的值 {score} 无效")
        if score < 0 or score > 1:
            raise ValueError(f"评分必须在 [0,1] 范围内，维度 {dim} 的值 {score} 超出范围")

    # P1-9: 权重归一化用副本，不修改原输入
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 0.001:
        # 自动归一化（使用副本）
        normalized_weights = {k: v / weight_sum for k, v in weights.items()}
        weights_to_use = normalized_weights
    else:
        # 权重和为1，使用副本避免修改原字典
        weights_to_use = dict(weights)

    # 计算加权得分
    weighted_scores = {}
    total = 0.0
    for dim in scores:
        w = weights_to_use.get(dim, 0)
        s = scores.get(dim, 0.5)
        ws = s * w
        weighted_scores[dim] = {
            "得分": round(s, 2),
            "权重": round(w, 4),
            "加权得分": round(ws, 4),
        }
        total += ws

    # 判读
    if total <= 0.30:
        verdict = "强利空，建议空头思路"
    elif total <= 0.45:
        verdict = "偏空，谨慎偏空操作"
    elif total <= 0.55:
        verdict = "中性，观望为主"
    elif total <= 0.70:
        verdict = "偏多，谨慎偏多操作"
    else:
        verdict = "强利多，建议多头思路"

    return {
        "维度明细": weighted_scores,
        "综合得分": round(total, 2),
        "多空研判": verdict,
    }


def generate_action_plan(
    total_score: float,
    dim11_data: dict = None,
    dim13_data: dict = None,
    dim14_data: dict = None,
    dim9_data: dict = None,
    volatility_type: str = "normal",
) -> dict:
    """
    基于综合得分和关键维度数据，生成操作方案

    Args:
        total_score: 综合得分 (0-1)
        dim11_data: 维度11(价格位置)数据，含:
            - support: 支撑位
            - resistance: 阻力位
            - ma20: 20日均线
            - boll_upper: 布林带上轨
            - boll_lower: 布林带下轨
            - low_20d: 20日最低价
            - high_20d: 20日最高价
        dim13_data: 维度13(资金面)数据，含:
            - position_change: 持仓量变化方向 ("增仓"/"减仓")
            - price_position_relation: 量价关系 ("增仓上涨"/"增仓下跌"/"减仓上涨"/"减仓下跌")
        dim14_data: 维度14(价差结构)数据，含:
            - spread_signal: 价差信号 ("backwardation"/"contango"/"中性")
            - spread_percentile: 价差历史分位数 (0-1)
        dim9_data: 维度9(基差)数据，含:
            - basis: 基差值 (正=现货升水, 负=现货贴水)
            - basis_signal: 基差信号 ("利多"/"利空"/"中性")
        volatility_type: 品种波动率类型 ("normal"/"high")

    Returns:
        操作方案字典
    """
    # 方向判断
    if total_score >= 0.55:
        direction = "多头"
    elif total_score <= 0.45:
        direction = "空头"
    else:
        direction = "观望"

    # 交叉确认计数
    confirmations = 0
    conflicts = 0
    confirmation_details = []

    # 资金面确认
    if dim13_data:
        price_pos = dim13_data.get("price_position_relation", "")
        if direction == "多头" and "增仓上涨" in price_pos:
            confirmations += 1
            confirmation_details.append(f"资金面确认: {price_pos}")
        elif direction == "空头" and "增仓下跌" in price_pos:
            confirmations += 1
            confirmation_details.append(f"资金面确认: {price_pos}")
        elif direction != "观望":
            conflicts += 1
            confirmation_details.append(f"资金面冲突: {price_pos}")

    # 价差结构确认
    if dim14_data:
        spread_signal = dim14_data.get("spread_signal", "")
        if direction == "多头" and spread_signal == "backwardation":
            confirmations += 1
            confirmation_details.append(f"价差结构确认: {spread_signal}(牛市结构)")
        elif direction == "空头" and spread_signal == "contango":
            confirmations += 1
            confirmation_details.append(f"价差结构确认: {spread_signal}(熊市结构)")
        elif direction != "观望":
            conflicts += 1
            confirmation_details.append(f"价差结构冲突: {spread_signal}")

    # 基差确认
    if dim9_data:
        basis_signal = dim9_data.get("basis_signal", "")
        if direction == "多头" and basis_signal == "利多":
            confirmations += 1
            confirmation_details.append(f"基差确认: {basis_signal}")
        elif direction == "空头" and basis_signal == "利空":
            confirmations += 1
            confirmation_details.append(f"基差确认: {basis_signal}")
        elif direction != "观望":
            conflicts += 1
            confirmation_details.append(f"基差冲突: {basis_signal}")

    # 开仓许可：需至少2条确认
    can_open = direction != "观望" and confirmations >= 2

    # 仓位建议
    score_deviation = abs(total_score - 0.5)
    if direction == "观望":
        position_pct = 0
    elif score_deviation < 0.10:
        position_pct = 10
    elif score_deviation < 0.20:
        position_pct = 15
    else:
        position_pct = 20

    # 维度冲突降仓
    if conflicts > 0 and can_open:
        position_pct = max(5, position_pct // 2)
        confirmation_details.append(f"⚠ 维度冲突({conflicts}项)，仓位减半至{position_pct}%")

    # 止损幅度
    stop_loss_pct = 5.0 if volatility_type == "high" else 3.0

    # 入场区间（基于维度11）
    entry_zone = "数据不足，无法确定"
    stop_loss_price = "数据不足"
    target1 = "数据不足"
    target2 = "数据不足"

    if dim11_data:
        support = dim11_data.get("support")
        resistance = dim11_data.get("resistance")
        low_20d = dim11_data.get("low_20d")
        high_20d = dim11_data.get("high_20d")
        boll_lower = dim11_data.get("boll_lower")
        boll_upper = dim11_data.get("boll_upper")

        if direction == "多头":
            entry_zone = f"{support} ~ {dim11_data.get('ma20', support)}"
            if low_20d:
                stop_loss_price = f"{low_20d} (20日低点)"
            target1 = f"{resistance or high_20d} (上方阻力)"
        elif direction == "空头":
            entry_zone = f"{dim11_data.get('ma20', resistance)} ~ {resistance}"
            if high_20d:
                stop_loss_price = f"{high_20d} (20日高点)"
            target1 = f"{support or low_20d} (下方支撑)"

        # 第二目标位（结合维度14和9）
        t2_parts = []
        if dim14_data and dim14_data.get("spread_percentile") is not None:
            pct = dim14_data["spread_percentile"]
            if direction == "多头" and pct < 0.2:
                t2_parts.append("价差处于极端低位(牛市结构强化)")
            elif direction == "空头" and pct > 0.8:
                t2_parts.append("价差处于极端高位(熊市结构强化)")
        if dim9_data and dim9_data.get("basis") is not None:
            t2_parts.append(f"基差均值回归位(当前基差{dim9_data['basis']})")
        target2 = "；".join(t2_parts) if t2_parts else target1

    # 数据不足检查
    data_sufficient = dim11_data is not None and dim13_data is not None
    if not data_sufficient and direction != "观望":
        can_open = False
        confirmation_details.append("⚠ 维度11/13数据不足，不建议开仓")

    return {
        "方向判断": {
            "建议方向": direction,
            "综合得分": total_score,
            "可否开仓": can_open,
            "确认信号": confirmation_details,
            "确认数": confirmations,
            "冲突数": conflicts,
        },
        "入场方案": {
            "入场区间": entry_zone,
            "触发条件": "价格触及入场区间 + 资金面确认" if can_open else "不满足开仓条件",
            "建议仓位": f"{position_pct}%" if can_open else "0%",
        },
        "止损方案": {
            "止损价位": stop_loss_price,
            "止损幅度": f"{stop_loss_pct}%",
            "止损后观察": "评估是否假突破，资金面未反转可考虑二次入场",
        },
        "止盈与加仓": {
            "第一目标位": target1,
            "第二目标位": target2,
            "加仓条件": f"盈利达{stop_loss_pct * 1.5:.1f}%后 + 资金面持续确认，加仓不超过首次仓位",
            "移动止损": "盈利达第一目标后止损移至成本价；达第二目标后移至第一目标位",
        },
    }


def default_weights(num_dims=14) -> dict:
    """
    返回默认等权重

    Args:
        num_dims: 维度数量，默认14（重构后维度体系）

    Returns:
        各维度权重字典
    """
    # 重构后的14维度名称
    dim_names = [
        "货币政策",      # 1
        "地缘政治",      # 2
        "产业政策",      # 3
        "关键经济指标",  # 4
        "市场情绪",      # 5
        "财政政策",      # 6
        "供应",          # 7
        "需求",          # 8
        "基差",          # 9
        "跨市场",        # 10
        "价格位置",  # 11（重构）
        "周期性",        # 12
        "资金面",        # 13（新增）
        "价差结构",      # 14（新增）
    ]
    w = 1.0 / num_dims
    return {f"{i+1}_{name}": round(w, 4) for i, name in enumerate(dim_names)}


def main():
    # 示例用法
    scores = {
        "1_货币政策": 0.60,
        "2_地缘政治": 0.50,
        "3_产业政策": 0.65,
        "4_关键经济指标": 0.55,
        "5_市场情绪": 0.45,
        "6_财政政策": 0.60,
        "7_供应": 0.40,
        "8_需求": 0.55,
        "9_基差": 0.50,
        "10_跨市场": 0.55,
        "11_价格位置": 0.50,
        "12_周期性": 0.45,
        "13_资金面": 0.55,
        "14_价差结构": 0.50,
    }

    weights = default_weights()
    result = calculate_weighted_score(scores, weights)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
