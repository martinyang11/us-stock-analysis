#!/usr/bin/env python3
"""
CANN 预训练样本生成器
生成10,000个合成样本用于模型预训练

X: 38维CA评分，涵盖多种市场状态
  - 极端利多（全维0.78-0.95）
  - 极端利空（全维0.05-0.22）
  - 偏多（全维0.58-0.78）
  - 偏空（全维0.22-0.42）
  - 中性（全维0.45-0.55）
  - 矛盾（部分偏高、部分偏低）
  - 趋势+反转混合
  - 基本面/技术面背离

Y: 38维加权平均分
  权重分三层：宏观(4维, 20%) + 产业(10维, 35%) + 技术(24维, 45%)
  每层内部按维度重要性进一步分配权重

用法: python generate_pretrain.py --output ../data/pretrain_samples.npz --n 10000
"""

import argparse
import json
import os
import sys

import numpy as np

# 项目路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..', '..')
sys.path.insert(0, _project_root)

# ============================================================
# 权重设计（38维）
# ============================================================

# 宏观层权重 (D1,D2,D4,D5) — 占总权重 20%
MACRO_WEIGHTS = np.array([
    0.060,  # D01 货币政策 — 影响全局定价锚
    0.040,  # D02 地缘政治 — 供给冲击
    0.050,  # D04 关键经济指标 — 需求预期
    0.050,  # D05 市场情绪 — 短期情绪放大器
])

# 产业层权重 (D3,D6,D7,D8,D9,D10,D11,D12,D13,D14) — 占 35%
INDUSTRIAL_WEIGHTS = np.array([
    0.050,  # D03 产业政策
    0.030,  # D06 财政政策
    0.070,  # D07 供应 — 商品最核心基本面
    0.050,  # D08 需求
    0.030,  # D09 基差
    0.030,  # D10 跨市场
    0.030,  # D11 价格位置
    0.020,  # D12 周期性
    0.030,  # D13 资金面
    0.010,  # D14 价差结构
])

# 技术面层权重 (T1-T24) — 占 45%
# 趋势方向/强度/动量最重，量仓/回撤次之，形态最轻
TECH_WEIGHTS = np.array([
    0.040,  # T01 趋势方向
    0.030,  # T02 趋势强度
    0.020,  # T03 动量
    0.020,  # T04 多空力量
    0.015,  # T05 波动率水平
    0.015,  # T06 波动率变化
    0.020,  # T07 量能强度
    0.025,  # T08 量价一致性
    0.015,  # T09 持仓变化率
    0.025,  # T10 量仓共振率 ★
    0.015,  # T11 均值偏离度
    0.015,  # T12 超买超卖
    0.015,  # T13 日内振幅
    0.030,  # T14 回撤深度 ★
    0.015,  # T15 锤子线
    0.015,  # T16 吞没形态
    0.015,  # T17 十字星
    0.015,  # T18 三连阳/阴
    0.015,  # T19 跳空
    0.015,  # T20 孕线
    0.015,  # T21 刺透形态
    0.015,  # T22 乌云盖顶
    0.020,  # T23 向上突破
    0.020,  # T24 向下突破
])

# 拼接为统一38维权重数组
# 基本面顺序: D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11 D12 D13 D14
# 技术面顺序: T1-T24
FULL_WEIGHTS = np.concatenate([
    [MACRO_WEIGHTS[0]],     # D01
    [MACRO_WEIGHTS[1]],     # D02
    [INDUSTRIAL_WEIGHTS[0]], # D03
    [MACRO_WEIGHTS[2]],     # D04
    [MACRO_WEIGHTS[3]],     # D05
    INDUSTRIAL_WEIGHTS[1:], # D06-D14
    TECH_WEIGHTS,           # T01-T24
])

# 归一化确保和为1
FULL_WEIGHTS = FULL_WEIGHTS / FULL_WEIGHTS.sum()

# 维度分组索引（用于矛盾样本生成）
FUNDAMENTAL_IDX = list(range(0, 14))    # D01-D14
TECHNICAL_IDX = list(range(14, 38))     # T01-T24
MACRO_IDX = [0, 1, 3, 4]               # D01,D02,D04,D05
INDUSTRIAL_IDX = [2, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 其余基本面


# ============================================================
# 样本生成
# ============================================================

def _clip_noise(dims, noise_scale=0.03):
    """添加小量高斯噪声并裁剪到[0,1]"""
    noise = np.random.randn(len(dims)) * noise_scale
    return np.clip(dims + noise, 0.0, 1.0)


def generate_samples(n=10000, seed=42):
    """生成预训练样本

    Returns:
        X: (n, 38) 特征矩阵
        y: (n,) 标签向量
        meta: list[dict] 每个样本的元信息（类型等）
    """
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 38), dtype=np.float64)
    meta_list = []

    # ---- 样本类型分配 ----
    types = [
        ('extreme_bullish', 1200),
        ('extreme_bearish', 1200),
        ('bullish', 1300),
        ('bearish', 1300),
        ('neutral', 2500),
        ('contradictory_fund_vs_tech', 800),
        ('contradictory_macro_vs_industrial', 600),
        ('trend_reversal', 600),
        ('mixed_categories', 500),
    ]

    idx = 0
    for type_name, count in types:
        for _ in range(count):
            if idx >= n:
                break

            if type_name == 'extreme_bullish':
                # 全维度 0.78-0.95，小幅扰动
                base = rng.uniform(0.78, 0.95, 38)
                X[idx] = _clip_noise(base, 0.02)

            elif type_name == 'extreme_bearish':
                base = rng.uniform(0.05, 0.22, 38)
                X[idx] = _clip_noise(base, 0.02)

            elif type_name == 'bullish':
                base = rng.uniform(0.58, 0.78, 38)
                X[idx] = _clip_noise(base, 0.04)

            elif type_name == 'bearish':
                base = rng.uniform(0.22, 0.42, 38)
                X[idx] = _clip_noise(base, 0.04)

            elif type_name == 'neutral':
                base = rng.uniform(0.45, 0.55, 38)
                X[idx] = _clip_noise(base, 0.03)

            elif type_name == 'contradictory_fund_vs_tech':
                # 基本面 vs 技术面 背离
                if rng.rand() > 0.5:
                    # 基本面利多 + 技术面利空
                    X[idx, FUNDAMENTAL_IDX] = rng.uniform(0.65, 0.90, 14)
                    X[idx, TECHNICAL_IDX] = rng.uniform(0.10, 0.35, 24)
                else:
                    # 基本面利空 + 技术面利多
                    X[idx, FUNDAMENTAL_IDX] = rng.uniform(0.10, 0.35, 14)
                    X[idx, TECHNICAL_IDX] = rng.uniform(0.65, 0.90, 24)
                X[idx] = _clip_noise(X[idx], 0.03)

            elif type_name == 'contradictory_macro_vs_industrial':
                # 宏观 vs 产业背离
                if rng.rand() > 0.5:
                    X[idx, MACRO_IDX] = rng.uniform(0.65, 0.90, 4)
                    X[idx, INDUSTRIAL_IDX] = rng.uniform(0.10, 0.35, 10)
                else:
                    X[idx, MACRO_IDX] = rng.uniform(0.10, 0.35, 4)
                    X[idx, INDUSTRIAL_IDX] = rng.uniform(0.65, 0.90, 10)
                # 技术面中性偏随机
                X[idx, TECHNICAL_IDX] = rng.uniform(0.35, 0.65, 24)
                X[idx] = _clip_noise(X[idx], 0.03)

            elif type_name == 'trend_reversal':
                # 模拟趋势反转：一些维度极端，另一些反向
                n_reverse = rng.randint(10, 25)
                reverse_idx = rng.choice(38, n_reverse, replace=False)
                non_reverse = [i for i in range(38) if i not in reverse_idx]
                # 主体偏多
                X[idx, non_reverse] = rng.uniform(0.60, 0.85, len(non_reverse))
                # 反转部分偏空
                X[idx, reverse_idx] = rng.uniform(0.15, 0.40, len(reverse_idx))
                X[idx] = _clip_noise(X[idx], 0.04)

            elif type_name == 'mixed_categories':
                # 各类别独立采样，模拟真实复杂市场
                groups = {
                    'macro': (MACRO_IDX, (0.20, 0.80)),
                    'industrial': (INDUSTRIAL_IDX, (0.20, 0.80)),
                    'tech_trend': (list(range(14, 19)), (0.15, 0.85)),     # T01-T04
                    'tech_volume': (list(range(19, 24)), (0.15, 0.85)),    # T05-T09
                    'tech_structure': (list(range(24, 28)), (0.15, 0.85)), # T10-T13
                    'tech_pattern': (list(range(28, 38)), (0.20, 0.80)),   # T14-T24
                }
                for dims, (lo, hi) in groups.values():
                    X[idx, dims] = rng.uniform(lo, hi, len(dims))
                X[idx] = _clip_noise(X[idx], 0.04)

            meta_list.append({'type': type_name, 'idx_in_type': _})
            idx += 1

    # 截断到实际生成数量
    X = X[:idx]
    n_actual = idx

    # ---- 计算 Y：加权平均 ----
    y = X @ FULL_WEIGHTS

    print(f"生成完成: {n_actual} 样本")
    print(f"  y 范围: [{y.min():.4f}, {y.max():.4f}]")
    print(f"  y 均值: {y.mean():.4f}, σ={y.std():.4f}")

    # 分布统计
    for type_name, _ in types:
        count = sum(1 for m in meta_list if m['type'] == type_name)
        if count > 0:
            type_y = np.array([y[i] for i, m in enumerate(meta_list) if m['type'] == type_name])
            print(f"  {type_name}: {count:4d}个, y∈[{type_y.min():.3f},{type_y.max():.3f}], "
                  f"均值={type_y.mean():.3f}")

    return X, y, meta_list


# ============================================================
# 保存 & 主入口
# ============================================================

def save_samples(X, y, meta_list, output_path):
    """保存为 npz + json 元信息"""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    np.savez(
        output_path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        weights=FULL_WEIGHTS.astype(np.float32),
    )
    print(f"样本已保存: {output_path} (X={X.shape}, y={y.shape})")

    # 保存元信息和权重JSON
    meta_path = output_path.replace('.npz', '_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({
            'n_samples': int(len(y)),
            'n_dims': 38,
            'y_range': [float(y.min()), float(y.max())],
            'y_mean': float(y.mean()),
            'y_std': float(y.std()),
            'weights': FULL_WEIGHTS.tolist(),
            'weight_design': {
                'macro_4dims': MACRO_WEIGHTS.tolist(),
                'industrial_10dims': INDUSTRIAL_WEIGHTS.tolist(),
                'tech_24dims': TECH_WEIGHTS.tolist(),
            },
            'type_distribution': {
                t: sum(1 for m in meta_list if m['type'] == t)
                for t in set(m['type'] for m in meta_list)
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"元信息已保存: {meta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CANN预训练样本生成')
    parser.add_argument('--output', default='../data/pretrain_samples.npz',
                        help='输出文件路径')
    parser.add_argument('--n', type=int, default=10000,
                        help='样本数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()

    output_path = os.path.join(_script_dir, args.output)
    X, y, meta = generate_samples(n=args.n, seed=args.seed)
    save_samples(X, y, meta, output_path)
