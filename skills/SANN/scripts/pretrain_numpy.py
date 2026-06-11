#!/usr/bin/env python3
"""
SANN 纯NumPy神经网络 — TradFi gTrade版 v4.1

核心适配：
- 品种数 35→56, Embedding(56,8)
- y值 sigmoid 系数 20 (适配股票波动率)
- 数据源 gTrade + yfinance (TradFi永续)
- gTrade 美股+ETF+商品+加密

架构：
Input(48) → Linear(48,48) + BN + ReLU + Drop(0.25) + Residual
          → Linear(48,32) + BN + ReLU + Drop(0.25) + ResidualProj
          → Linear(32,16) + BN + ReLU + Drop(0.25) + ResidualProj
          → Linear(16,8)  + BN + ReLU + Drop(0.25) + ResidualProj
          → Linear(8,1) → Sigmoid → y ∈ [0,1]
"""

import os
import sys
import csv
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('SANN.numpy')

# ---- 品种映射 (gTrade 56个TradFi品种) ----
NUM_VARIETIES = 56
EMBEDDING_DIM = 8
INPUT_SCORES = 38  # 14基本面 + 24技术面
HIDDEN_DIMS = [48, 32, 16, 8]  # 4层
TOTAL_INPUT = INPUT_SCORES + 2 + EMBEDDING_DIM  # 38 + 2(month) + 8(embed) = 48


# ============================================================
# 激活函数
# ============================================================
def relu(x): return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1 / (1 + np.exp(-x))

def relu_derivative(x): return (x > 0).astype(np.float64)


# ============================================================
# 模型定义
# ============================================================
class NumpySANNModel:
    """4层残差MLP + Embedding + BatchNorm"""

    def __init__(self, hidden_dims=None, embedding_dim=8, input_scores=38):
        self.hidden_dims = hidden_dims or HIDDEN_DIMS
        self.embedding_dim = embedding_dim
        self.input_scores = input_scores
        self.num_varieties = NUM_VARIETIES
        self.total_input = input_scores + 2 + embedding_dim  # 48

        self.params = {}
        self.bn_running = {}  # BatchNorm 运行统计
        self._init_params()

    def _init_params(self):
        """Xavier初始化"""
        d = self.total_input
        dims = self.hidden_dims

        # Embedding层
        self.params['embedding'] = np.random.randn(NUM_VARIETIES, self.embedding_dim) * 0.01

        # 隐藏层
        for i, out_dim in enumerate(dims):
            in_dim = d if i == 0 else dims[i-1]
            scale = np.sqrt(2.0 / (in_dim + out_dim))
            self.params[f'fc{i+1}.W'] = np.random.randn(in_dim, out_dim) * scale
            self.params[f'fc{i+1}.b'] = np.zeros(out_dim)
            self.params[f'bn{i+1}.gamma'] = np.ones(out_dim)
            self.params[f'bn{i+1}.beta'] = np.zeros(out_dim)
            self.bn_running[f'bn{i+1}.mean'] = np.zeros(out_dim)
            self.bn_running[f'bn{i+1}.var'] = np.ones(out_dim)

            # 残差投影（维度不匹配时）
            if in_dim != out_dim:
                scale_p = np.sqrt(2.0 / (in_dim + out_dim))
                self.params[f'res_proj.{i+1}.W'] = np.random.randn(in_dim, out_dim) * scale_p
                self.params[f'res_proj.{i+1}.b'] = np.zeros(out_dim)

        # 输出层
        last_dim = dims[-1]  # 8
        scale_o = np.sqrt(2.0 / (last_dim + 1))
        self.params['output.W'] = np.random.randn(last_dim, 1) * scale_o
        self.params['output.b'] = np.zeros(1)

    def forward(self, scores, month, variety_id, training=True):
        """前向传播

        Args:
            scores: (batch_size, 38) 或 (38,) — CA评分
            month: int 或 (batch_size,) — 月份
            variety_id: int 或 (batch_size,) — 币种ID
            training: 是否训练模式（影响Dropout/BN）

        Returns:
            y_pred: (batch_size, 1) 或 (1,)
        """
        # 输入处理
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)
        batch_size = scores.shape[0]

        # 月份编码
        if isinstance(month, (int, float, np.integer, np.floating)):
            month_arr = np.full(batch_size, month)
        else:
            month_arr = np.array(month)

        month_sin = np.sin(2 * np.pi * month_arr / 12).reshape(-1, 1)
        month_cos = np.cos(2 * np.pi * month_arr / 12).reshape(-1, 1)

        # 币种Embedding
        if isinstance(variety_id, (int, np.integer)):
            vid_arr = np.full(batch_size, variety_id, dtype=int)
        else:
            vid_arr = np.array(variety_id, dtype=int)
        embed = self.params['embedding'][vid_arr]  # (batch, 8)

        # 拼接输入
        x = np.hstack([scores, month_sin, month_cos, embed])  # (batch, 48)

        cache = {'x': x}

        # 4层残差
        for i, out_dim in enumerate(self.hidden_dims):
            in_dim = x.shape[1]
            layer_key = f'fc{i+1}'
            bn_key = f'bn{i+1}'

            # Linear
            z = x @ self.params[f'{layer_key}.W'] + self.params[f'{layer_key}.b']

            # BatchNorm
            if training:
                mean = z.mean(axis=0, keepdims=True)
                var = z.var(axis=0, keepdims=True) + 1e-5
                self.bn_running[f'{bn_key}.mean'] = 0.9 * self.bn_running[f'{bn_key}.mean'] + 0.1 * mean[0]
                self.bn_running[f'{bn_key}.var'] = 0.9 * self.bn_running[f'{bn_key}.var'] + 0.1 * var[0]
            else:
                mean = self.bn_running[f'{bn_key}.mean'].reshape(1, -1)
                var = self.bn_running[f'{bn_key}.var'].reshape(1, -1) + 1e-5

            z_norm = (z - mean) / np.sqrt(var)
            z_bn = z_norm * self.params[f'{bn_key}.gamma'] + self.params[f'{bn_key}.beta']

            # ReLU
            a = relu(z_bn)

            # Dropout
            if training:
                mask = (np.random.rand(*a.shape) > 0.25).astype(np.float64) / 0.75
                a = a * mask
                cache[f'dropout_mask_{i+1}'] = mask

            # 残差连接
            if in_dim != out_dim:
                residual = x @ self.params[f'res_proj.{i+1}.W'] + self.params[f'res_proj.{i+1}.b']
            else:
                residual = x

            x = a + residual

            cache[f'layer_{i+1}'] = {'z': z, 'z_bn': z_bn, 'a': a, 'residual': residual,
                                       'mean': mean, 'var': var, 'x_input': x}

        # 输出层
        y_logit = x @ self.params['output.W'] + self.params['output.b']
        y_pred = sigmoid(y_logit)

        cache['output'] = {'logit': y_logit, 'y_pred': y_pred, 'x_last': x}
        return y_pred, cache

    def backward(self, cache, y_true):
        """反向传播，计算梯度"""
        grads = {}
        batch_size = y_true.shape[0]

        # 输出层梯度: dL/dy * y*(1-y) 对于sigmoid+MSE
        y_pred = cache['output']['y_pred']
        x_last = cache['output']['x_last']

        # MSE: dL/dy = (y_pred - y_true) / batch_size
        dy = (y_pred - y_true) / batch_size
        # Sigmoid derivative: y * (1-y)
        dz_out = dy * y_pred * (1 - y_pred)

        grads['output.W'] = x_last.T @ dz_out
        grads['output.b'] = dz_out.sum(axis=0)

        # 反向传播通过各层
        d_prev = dz_out @ self.params['output.W'].T  # (batch, 8)

        for i in range(len(self.hidden_dims), 0, -1):
            layer_cache = cache[f'layer_{i}']
            a = layer_cache['a']
            z_bn = layer_cache['z_bn']
            x_input = layer_cache['x_input']
            mean = layer_cache['mean']
            var = layer_cache['var']

            # 残差梯度通过
            d_residual = d_prev.copy()
            d_after_relu = d_prev.copy()

            # ReLU backward
            d_zbn = d_after_relu * relu_derivative(z_bn)

            # Dropout backward
            if f'dropout_mask_{i}' in cache:
                d_zbn = d_zbn * cache[f'dropout_mask_{i}']

            # BatchNorm backward
            N = batch_size
            gamma = self.params[f'bn{i}.gamma']
            std_inv = 1.0 / np.sqrt(var + 1e-5)
            x_centered = layer_cache['z'] - mean

            d_gamma = (d_zbn * x_centered * std_inv).sum(axis=0)
            d_beta = d_zbn.sum(axis=0)
            d_z = (1. / N) * gamma * std_inv * (
                N * d_zbn - d_zbn.sum(axis=0) - x_centered * std_inv**2 * (d_zbn * x_centered).sum(axis=0)
            )

            grads[f'bn{i}.gamma'] = d_gamma
            grads[f'bn{i}.beta'] = d_beta

            # Linear backward
            prev_x = cache[f'layer_{i-1}']['x_input'] if i > 1 else cache['x']
            grads[f'fc{i}.W'] = prev_x.T @ d_z
            grads[f'fc{i}.b'] = d_z.sum(axis=0)

            # 残差投影
            in_dim = prev_x.shape[1]
            out_dim = self.hidden_dims[i-1]
            if in_dim != out_dim:
                grads[f'res_proj.{i}.W'] = prev_x.T @ d_residual
                grads[f'res_proj.{i}.b'] = d_residual.sum(axis=0)
                d_prev = d_z @ self.params[f'fc{i}.W'].T + d_residual @ self.params[f'res_proj.{i}.W'].T
            else:
                d_prev = d_z @ self.params[f'fc{i}.W'].T + d_residual

        # Embedding梯度（简化：只更新用到的币种）
        # 实际实现中通过索引累积，此处简化处理

        return grads


# ============================================================
# Adam优化器
# ============================================================
class AdamOptimizer:
    def __init__(self, lr=5e-5, beta1=0.9, beta2=0.999, eps=1e-8,
                 clip_norm=1.0, embedding_lr_mult=5.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_norm = clip_norm
        self.embedding_lr_mult = embedding_lr_mult
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, model: NumpySANNModel, grads: dict):
        self.t += 1
        # 全局梯度裁剪
        total_norm = 0
        for g in grads.values():
            total_norm += float(np.sum(g ** 2))
        total_norm = np.sqrt(total_norm)

        for key in model.params:
            if key not in grads:
                continue
            g = grads[key]
            if total_norm > self.clip_norm:
                g = g * (self.clip_norm / total_norm)

            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            lr = self.lr
            if key == 'embedding':
                lr *= self.embedding_lr_mult

            model.params[key] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# 数据加载
# ============================================================
def load_csv_samples(csv_path: str, filter_invalid: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 historical_samples.csv

    Returns:
        features: (N, 38) — 14CA + 24Tech
        months: (N,)
        variety_ids: (N,)
        y: (N,)
        raw_changes: (N,)
    """
    if not os.path.exists(csv_path):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    features = []
    months = []
    vids = []
    ys = []
    changes = []

    for row in rows:
        # 读取14CA+24Tech
        dims = []
        for i in range(1, 15):
            dims.append(float(row.get(f'dim{i}', '-1')))
        for i in range(1, 25):
            dims.append(float(row.get(f'tech{i}', row.get(f'dim{i+14}', '-1'))))  # 兼容新/旧格式

        raw_change = float(row.get('raw_change', '0.0'))

        if filter_invalid:
            # 过滤：dim为-1 或 raw_change=0
            if any(d < 0 for d in dims[:14]):  # 只检查CA维度
                continue
            if abs(raw_change) < 1e-8:
                continue

        cid = int(row.get('crypto_id', row.get('variety_id', '0')))
        month = int(row.get('month', '1'))

        features.append(dims)
        months.append(month)
        vids.append(cid)
        ys.append(float(row.get('y', '0.5')))
        changes.append(raw_change)

    return (np.array(features, dtype=np.float64),
            np.array(months, dtype=np.int32),
            np.array(vids, dtype=np.int32),
            np.array(ys, dtype=np.float64),
            np.array(changes, dtype=np.float64))


# ============================================================
# 训练
# ============================================================
def run_daily_training_numpy(data_dir: str, verbose: bool = True) -> Optional[NumpySANNModel]:
    """每日微调"""
    csv_path = os.path.join(data_dir, 'historical_samples.csv')
    features, months, vids, ys, _ = load_csv_samples(csv_path, filter_invalid=True)

    if len(features) < 25:
        if verbose:
            print(f'  ⚠️ 有效样本不足({len(features)}<25)，跳过训练')
        return None

    # 时序划分
    n = len(features)
    split = int(n * 0.8)
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    # 初始化模型
    model = NumpySANNModel()
    optimizer = AdamOptimizer()

    best_val_loss = float('inf')
    patience_counter = 0
    batch_size = min(32, split)

    for epoch in range(100):
        # Shuffle训练集
        np.random.shuffle(train_idx)

        # Mini-batch训练
        train_losses = []
        for start in range(0, split, batch_size):
            end = min(start + batch_size, split)
            idx = train_idx[start:end]

            batch_features = features[idx]
            batch_months = months[idx]
            batch_vids = vids[idx]
            batch_y = ys[idx].reshape(-1, 1)

            y_pred, cache = model.forward(batch_features, batch_months, batch_vids, training=True)
            grads = model.backward(cache, batch_y)
            optimizer.step(model, grads)

            loss = float(np.mean((y_pred - batch_y) ** 2))
            train_losses.append(loss)

        # 验证
        val_features = features[val_idx]
        val_y = ys[val_idx].reshape(-1, 1)
        y_val, _ = model.forward(val_features, months[val_idx], vids[val_idx], training=False)
        val_loss = float(np.mean((y_val - val_y) ** 2))

        if verbose and epoch % 20 == 0:
            print(f'  Epoch {epoch}: train_loss={np.mean(train_losses):.6f}  val_loss={val_loss:.6f}')

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model, os.path.join(data_dir, 'model_weights.npz'))
        else:
            patience_counter += 1
            if patience_counter >= 15:
                if verbose:
                    print(f'  早停: val_loss未改善 {patience_counter}轮')
                break

    if verbose:
        print(f'  训练完成: best_val_loss={best_val_loss:.6f}, 样本={n}')

    return model


# ============================================================
# 推理
# ============================================================
def predict_single(model: NumpySANNModel, dim_scores, month, variety_id,
                   tech_scores=None) -> float:
    """单样本推理

    Args:
        model: NumpySANNModel
        dim_scores: 38维 (14CA+24Tech) 或 14维(需要tech_scores)
        month: int
        variety_id: int (0-49)
        tech_scores: 24维技术面评分（当dim_scores只有14维时）

    Returns:
        cann_score: float [0,1]
    """
    if model is None:
        return 0.5

    dims = np.array(dim_scores, dtype=np.float64)
    if len(dims) == 14 and tech_scores is not None:
        tech = np.array(tech_scores, dtype=np.float64)
        dims = np.concatenate([dims, tech])

    if len(dims) != 38:
        return 0.5

    y_pred, _ = model.forward(dims, month, variety_id, training=False)
    return float(np.clip(y_pred[0, 0], 0.001, 0.999))


# ============================================================
# 模型保存/加载
# ============================================================
def save_model(model: NumpySANNModel, path: str):
    """保存模型权重"""
    save_dict = {}
    for key, val in model.params.items():
        save_dict[key] = val
    for key, val in model.bn_running.items():
        save_dict[f'_bn_{key}'] = val
    save_dict['_hidden_dims'] = np.array(model.hidden_dims)
    save_dict['_input_scores'] = np.array([model.input_scores])
    save_dict['_version'] = np.array([4, 0])
    save_dict['_num_varieties'] = np.array([NUM_VARIETIES])
    np.savez_compressed(path, **save_dict)


def load_pretrained_model(data_dir: str) -> Tuple[Optional[NumpySANNModel], str]:
    """加载预训练模型权重"""
    # 优先加载 model_weights.npz，否则加载预训练权重
    for fname in ['model_weights.npz', 'model_weights_pretrained.npz']:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            try:
                data = np.load(path, allow_pickle=True)
                model = NumpySANNModel()
                # 恢复参数
                for key in model.params:
                    if key in data:
                        model.params[key] = data[key]
                # 恢复BN统计
                for key in model.bn_running:
                    bn_key = f'_bn_{key}'
                    if bn_key in data:
                        model.bn_running[key] = data[bn_key]
                logger.info(f'模型加载: {path}')
                return model, path
            except Exception as e:
                logger.warning(f'加载失败 {path}: {e}')

    return None, ''


# ============================================================
# 预训练数据生成（模拟历史样本）
# ============================================================
def generate_pretrain_samples(data_dir: str, n_samples: int = 500):
    """生成预训练样本供冷启动使用

    使用简单的启发式规则生成模拟样本，帮助模型获得初始权重。
    真实数据积累后将取代这些模拟样本。
    """
    print(f'生成 {n_samples} 条预训练样本...')

    np.random.seed(42)
    fieldnames = ['date', 'crypto_id', 'crypto_name', 'month'] + \
                 [f'dim{i}' for i in range(1, 15)] + \
                 [f'tech{i}' for i in range(1, 25)] + \
                 ['y', 'raw_change']

    # 用简单关联规则：资金费率(D9)+OI(D13)+价格位置(D11)与次日涨跌的已知相关性
    samples = []
    base_date = '2026-01-01'

    from datetime import timedelta
    for i in range(n_samples):
        dt = (np.datetime64(base_date) + np.timedelta64(i % 200, 'D'))
        cid = np.random.randint(0, NUM_VARIETIES)
        month = int(str(dt).split('-')[1])

        # 生成有意义的CA评分（带噪声）
        base_score = 0.5 + np.random.randn() * 0.15
        dims = np.clip(base_score + np.random.randn(14) * 0.1, 0.05, 0.95)

        # 技术面（与CA相关但带独立噪声）
        techs = np.clip(dims + np.random.randn(14) * 0.08, 0.05, 0.95)
        # 补足24维
        while len(techs) < 24:
            techs = np.append(techs, np.clip(0.5 + np.random.randn() * 0.15, 0.05, 0.95))

        # y值与评分相关（MSE目标）
        ca_mean = float(np.mean(dims))
        y_true = np.clip(ca_mean + np.random.randn() * 0.1, 0.05, 0.95)
        raw_change = -np.log(1/y_true - 1) / 20  # 反推sigmoid

        sample = {
            'date': str(dt),
            'crypto_id': cid,
            'crypto_name': f'模拟币种{cid}',
            'month': month,
            'y': f'{y_true:.6f}',
            'raw_change': f'{raw_change:.6f}',
        }
        for j in range(1, 15):
            sample[f'dim{j}'] = f'{dims[j-1]:.4f}'
        for j in range(1, 25):
            sample[f'tech{j}'] = f'{techs[min(j-1, 13)]:.4f}'

        samples.append(sample)

    # 保存
    path = os.path.join(data_dir, 'historical_samples.csv')
    os.makedirs(data_dir, exist_ok=True)

    # 如果已存在真实数据，追加到末尾
    existing = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing = [r for r in reader if float(r.get('raw_change', '0')) != 0]

    all_rows = existing + samples
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'✅ 预训练样本生成: {len(samples)}条 (总{len(all_rows)}条) → {path}')
    return len(all_rows)


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='SANN numPy 神经网络 (Crypto)')
    parser.add_argument('--data-dir', default='../data')
    parser.add_argument('--generate-pretrain', action='store_true',
                       help='生成预训练样本')
    parser.add_argument('--train', action='store_true',
                       help='执行训练')
    parser.add_argument('--predict', type=str, default=None,
                       help='推理单个币种 (symbol, 如BTCUSDT)')

    args = parser.parse_args()

    if args.generate_pretrain:
        generate_pretrain_samples(args.data_dir)

    elif args.train:
        run_daily_training_numpy(args.data_dir)

    elif args.predict:
        model, _ = load_pretrained_model(args.data_dir)
        if model:
            # 模拟CA评分
            dummy_scores = np.full(38, 0.55)
            score = predict_single(model, dummy_scores, 6, 0)
            print(f'{args.predict}: SANN={score:.4f}')
        else:
            print('无可用模型')

    else:
        print("用法: python pretrain_numpy.py --generate-pretrain|--train|--predict BTCUSDT")
