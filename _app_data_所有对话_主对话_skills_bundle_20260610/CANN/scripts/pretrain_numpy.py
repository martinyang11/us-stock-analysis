#!/usr/bin/env python3
"""
CANN 纯NumPy实现 v4.3
架构：38维CA统一评分 + 2维月份编码 + 8维品种Embedding = 48维输入

模型结构：
- 输入：48维 (38 CA评分 + 2月份 + 8品种Embedding)
- 隐藏层：[48, 32, 16, 8]，每层 Linear → BatchNorm → ReLU → Dropout(0.25) + 残差连接
- 输出：1维 Sigmoid

损失：MSE_WEIGHT*MSE + BCE_WEIGHT*软标签BCE
优化器：Adam (手写NumPy实现，支持分层学习率、warmup和cosine decay)

当前训练配置：MSE_WEIGHT=1.0, BCE_WEIGHT=0.0
- BCE关闭原因：软标签BCE梯度量级是MSE的100+倍，导致模型不收敛

v4.3变更日志（vs v4.2）：
- [P1] 预训练阶段：合成数据10,000样本初始化权重，再真实数据微调
- [P1] load_pretrain_samples(): 从npz加载合成样本（随机品种ID+月份）
- [P1] pretrain_numpy(): 50 epoch预训练，lr=0.001，纯MSE
- [P1] run_daily_training_numpy(): Phase 1自动预训练 + Phase 2微调
- [P1] 所有训练路径强制预训练先行：daily/finetune/回退路径均检查预训练权重
- [P2] 新增 --action pretrain 和 --pretrain-epochs CLI参数
- [Sync] T10维度名 "量仓背离" → "量仓共振率"（v4.6四态加权已生效）

v4.2变更日志（vs v4.0）：
- [架构] 技术面归入CA：CA输出38维统一评分（14基本面+24技术面），CANN直接消费
- [架构] 移除tech_scores参数：forward/backward/predict_single/predict_numpy统一走38维
- [架构] 砍掉α混合：CatTrader纯CANN评分决策，CA综合分仅作参考
- [输入] INPUT_TECH常量删除，INPUT_SCORES=38，INPUT_TOTAL_SCORES=38
- [CSV] load_csv_samples直接读dim1-dim38，不再兼容旧tech1-tech24格式

v3.2变更日志（vs v3.1）：
- [P1] 验证集分层划分：按品种分组比例抽取
- [P1] 品种差异化p_range：默认±10%，品种差异化范围
- [P1] Embedding梯度提取防御：forward记录emb_slice位置
- [P1] 学习率warmup+cosine decay
- [P1] load_weights字段缺失检查

v3.1变更日志（vs v3.0）：
- [P0] BCE梯度自适应归一化
- [P0] eval模式BN反向传播修复
- [P0] 权重键名NumPy化
- [P0] load_pretrained_model路径修复
"""

import csv
import json
import logging
import math
import os
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# 添加项目路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '..', '..', '..')
sys.path.insert(0, _project_root)
from skills.common.variety_list import (
    VARIETY_NAMES, NUM_VARIETIES, DIM_NAMES, NUM_DIMENSIONS,
    TOTAL_INPUT_DIMENSIONS
)
from skills.common.price_utils import normalize_price_change, VARIETY_P_RANGES, DEFAULT_P_MIN, DEFAULT_P_MAX

# ============================================================
# 常量
# ============================================================
INPUT_SCORES = NUM_DIMENSIONS          # 38 CA统一评分（14基本面+24技术面）
INPUT_TOTAL_SCORES = TOTAL_INPUT_DIMENSIONS  # 38
EMBEDDING_DIM = 8
MONTH_FEATURES = 2
EFFECTIVE_INPUT = INPUT_TOTAL_SCORES + MONTH_FEATURES + EMBEDDING_DIM  # 48
HIDDEN_DIMS = [48, 32, 16, 8]
DROPOUT_RATE = 0.25
# DEFAULT_P_MIN/P_MAX 和 VARIETY_P_RANGES 已移至 skills.common.price_utils 统一管理
MSE_WEIGHT = 1.0
BCE_WEIGHT = 0.0
BN_MOMENTUM = 0.1
BN_EPS = 1e-5


# ============================================================
# 工具函数
# ============================================================
def sigmoid(x):
    """数值稳定的sigmoid"""
    pos_mask = x >= 0
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[~pos_mask] = np.exp(x[~pos_mask])
    top = np.ones_like(x)
    top[~pos_mask] = z[~pos_mask]
    return top / (1.0 + z)


def relu(x):
    return np.maximum(0, x)


def soft_label(target, gain=3.0):
    return sigmoid(gain * (target - 0.5))


# normalize_price_change 已移至 skills.common.price_utils 统一管理


# ============================================================
# Adam优化器
# ============================================================
class AdamOptimizer:
    """NumPy Adam，支持分层学习率、warmup和cosine decay"""

    def __init__(self, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 warmup_steps=0, total_steps=0):
        self.lr = lr
        self.base_lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        self.param_lr_scale = {}
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def set_lr_scale(self, param_name, scale):
        self.param_lr_scale[param_name] = scale

    def _get_current_lr(self):
        if self.warmup_steps > 0 and self.t <= self.warmup_steps:
            return self.base_lr * self.t / self.warmup_steps
        if self.total_steps > 0 and self.t <= self.total_steps:
            progress = (self.t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * 0.01

    def step(self, grads_dict):
        self.t += 1
        self.lr = self._get_current_lr()
        # v4.1: 梯度裁剪（clip_norm=1.0），防止梯度爆炸
        CLIP_GRAD_NORM = 1.0
        all_grads = []
        for name, (param_ref, grad) in grads_dict.items():
            if grad is None:
                continue
            all_grads.append(grad.ravel())
        if all_grads:
            global_grad_norm = np.linalg.norm(np.concatenate(all_grads))
            clip_coef = 1.0
            if global_grad_norm > CLIP_GRAD_NORM:
                clip_coef = CLIP_GRAD_NORM / global_grad_norm
        else:
            clip_coef = 1.0
        for name, (param_ref, grad) in grads_dict.items():
            if grad is None:
                continue
            grad = grad * clip_coef  # 应用梯度裁剪
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param_ref
            if name not in self.m:
                self.m[name] = np.zeros_like(grad)
                self.v[name] = np.zeros_like(grad)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            lr_scale = self.param_lr_scale.get(name, 1.0)
            param_ref -= self.lr * lr_scale * m_hat / (np.sqrt(v_hat) + self.eps)


# ============================================================
# NumPy MLP 模型 v4.0
# ============================================================
class NumpyCANNModel:
    """纯NumPy CANN模型 v4.2: [48,32,16,8]+残差+完整BN+Adam+软标签

    输入: 38维CA统一评分 + 2维月份(sin/cos) + 8维品种Embedding = 48维
    """

    def __init__(self, num_varieties=NUM_VARIETIES, embedding_dim=EMBEDDING_DIM,
                 hidden_dims=HIDDEN_DIMS, dropout=DROPOUT_RATE, seed=42,
                 input_scores=INPUT_TOTAL_SCORES):
        rng = np.random.RandomState(seed)
        self.dropout_rate = dropout
        self.training = False  # 默认eval模式，训练时显式设置True
        self.num_varieties = num_varieties
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.rng = rng
        self.input_scores = input_scores  # 38 CA统一评分
        self.effective_input = input_scores + MONTH_FEATURES + embedding_dim  # 48

        # Embedding
        self.embedding = rng.randn(num_varieties, embedding_dim) * 0.1

        # 全连接层 + BN + 残差投影
        self.layers = []
        self.residual_projections = []
        in_dim = self.effective_input
        for h_dim in hidden_dims:
            W = rng.randn(in_dim, h_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros(h_dim)
            self.layers.append({
                'W': W, 'b': b,
                'bn_gamma': np.ones(h_dim), 'bn_beta': np.zeros(h_dim),
                'bn_running_mean': np.zeros(h_dim), 'bn_running_var': np.ones(h_dim),
            })
            if in_dim != h_dim:
                self.residual_projections.append({
                    'W': rng.randn(in_dim, h_dim) * np.sqrt(2.0 / in_dim),
                    'b': np.zeros(h_dim)
                })
            else:
                self.residual_projections.append(None)
            in_dim = h_dim

        self.output_W = rng.randn(in_dim, 1) * np.sqrt(2.0 / in_dim)
        self.output_b = np.zeros(1)

        # Adam
        self.optimizer = AdamOptimizer(lr=0.001)
        # v4.1: Embedding分层学习率——scale=5.0（相对于主网络lr）
        # 原scale=1.0问题：cosine decay后lr降至0.01x，Embedding几乎停止更新
        # scale=5.0确保Embedding在训练后期仍有有效更新量（5x * 0.01x = 0.05x初始lr）
        for vid in range(num_varieties):
            self.optimizer.set_lr_scale(f'emb_{vid}', 5.0)

    def _get_layer_output(self, i):
        """获取第i层残差连接后的输出"""
        identity = self.cache['residual_identity'][i]
        a = self.cache['post_relu'][i].copy()
        mask = self.cache['dropout_masks'][i]
        if mask is not None:
            a = a * mask
        proj = self.residual_projections[i]
        if proj is not None:
            projected = identity @ proj['W'] + proj['b']
            return a + projected
        else:
            return a + identity

    def forward(self, scores, month_feat, variety_ids):
        """前向传播（含残差连接）

        Args:
            scores: CA统一38维评分 (batch, 38)
            month_feat: 月份特征 (batch, 2)
            variety_ids: 品种ID (batch,)
        """
        # v4.2: 输入校验
        if not isinstance(scores, np.ndarray) or scores.ndim != 2:
            raise ValueError(f"scores必须是2维numpy数组，当前type={type(scores).__name__}, ndim={getattr(scores, 'ndim', 'N/A')}")
        if scores.shape[1] != INPUT_TOTAL_SCORES:
            raise ValueError(f"scores第2维必须是{INPUT_TOTAL_SCORES}，当前={scores.shape[1]}")
        if not isinstance(variety_ids, (np.ndarray, list)):
            raise ValueError(f"variety_ids必须是numpy数组或列表，当前type={type(variety_ids).__name__}")
        if len(variety_ids) != scores.shape[0]:
            raise ValueError(f"variety_ids长度({len(variety_ids)})与scores批次大小({scores.shape[0]})不匹配")
        vid_arr = np.array(variety_ids)
        if np.any((vid_arr < 0) | (vid_arr >= self.num_varieties)):
            raise ValueError(f"variety_ids超出范围[0, {self.num_varieties})，当前范围=[{vid_arr.min()}, {vid_arr.max()}]")
        # v4.1: NaN/Inf检查，防止异常值污染模型
        if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
            nan_count = np.isnan(scores).sum()
            inf_count = np.isinf(scores).sum()
            logger.warning(f"scores包含NaN({nan_count}个)/Inf({inf_count}个)，自动替换为中性值")
            scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)

        emb = self.embedding[variety_ids]
        x = np.concatenate([scores, month_feat, emb], axis=1)

        # 记录embedding位置
        emb_start = INPUT_TOTAL_SCORES + MONTH_FEATURES  # 38 + 2 = 40
        emb_end = emb_start + self.embedding_dim           # 40 + 8 = 48

        self.cache = {'input': x, 'pre_bn': [], 'bn_norm': [], 'post_bn': [],
                      'post_relu': [], 'dropout_masks': [], 'residual_identity': [],
                      '_emb_slice': (emb_start, emb_end)}

        for i, layer in enumerate(self.layers):
            self.cache['residual_identity'].append(x)
            z = x @ layer['W'] + layer['b']
            self.cache['pre_bn'].append(z)

            # BatchNorm
            if self.training:
                batch_mean = np.mean(z, axis=0)
                batch_var = np.var(z, axis=0)
                layer['bn_running_mean'] = (1 - BN_MOMENTUM) * layer['bn_running_mean'] + BN_MOMENTUM * batch_mean
                layer['bn_running_var'] = (1 - BN_MOMENTUM) * layer['bn_running_var'] + BN_MOMENTUM * batch_var
                z_norm = (z - batch_mean) / np.sqrt(batch_var + BN_EPS)
            else:
                z_norm = (z - layer['bn_running_mean']) / np.sqrt(layer['bn_running_var'] + BN_EPS)

            self.cache['bn_norm'].append(z_norm)
            z_scaled = layer['bn_gamma'] * z_norm + layer['bn_beta']
            self.cache['post_bn'].append(z_scaled)

            a = relu(z_scaled)
            self.cache['post_relu'].append(a)

            if self.training and self.dropout_rate > 0:
                mask = (self.rng.rand(*a.shape) > self.dropout_rate).astype(np.float64) / (1 - self.dropout_rate)
                a = a * mask
                self.cache['dropout_masks'].append(mask)
            else:
                self.cache['dropout_masks'].append(None)

            # 残差连接
            proj = self.residual_projections[i]
            if proj is not None:
                x = a + self.cache['residual_identity'][-1] @ proj['W'] + proj['b']
            else:
                x = a + self.cache['residual_identity'][-1]

        out = x @ self.output_W + self.output_b
        pred = sigmoid(out.flatten())
        return pred

    def compute_loss(self, pred, target):
        mse = np.mean((pred - target) ** 2)
        soft_t = soft_label(target, gain=3.0)  # v4.1: 与soft_label默认值一致
        eps = 1e-7
        pred_c = np.clip(pred, eps, 1 - eps)
        bce = -np.mean(soft_t * np.log(pred_c) + (1 - soft_t) * np.log(1 - pred_c))
        return MSE_WEIGHT * mse + BCE_WEIGHT * bce, mse, bce

    def _bn_backward(self, dx, i):
        layer = self.layers[i]
        z_norm = self.cache['bn_norm'][i]
        pre_bn = self.cache['pre_bn'][i]

        d_bn_gamma = np.sum(dx * z_norm, axis=0)
        d_bn_beta = np.sum(dx, axis=0)
        dx_norm = dx * layer['bn_gamma']

        if self.training:
            batch_size_bn = pre_bn.shape[0]
            batch_var = np.var(pre_bn, axis=0) + BN_EPS
            batch_mean = np.mean(pre_bn, axis=0)
            d_var = np.sum(dx_norm * (pre_bn - batch_mean) * -0.5 * batch_var ** (-1.5), axis=0)
            d_mean = np.sum(dx_norm * -1.0 / np.sqrt(batch_var), axis=0) + d_var * np.mean(-2.0 * (pre_bn - batch_mean), axis=0)
            dz = dx_norm / np.sqrt(batch_var) + d_var * 2.0 * (pre_bn - batch_mean) / batch_size_bn + d_mean / batch_size_bn
        else:
            running_std = np.sqrt(layer['bn_running_var'] + BN_EPS)
            dz = dx_norm / running_std

        return d_bn_gamma, d_bn_beta, dz

    def _compute_grads(self, d_pred):
        batch_size = len(d_pred)
        grads = {}
        d_out = (d_pred * self.cache.get('_preds', np.zeros_like(d_pred)) *
                 (1 - self.cache.get('_preds', np.zeros_like(d_pred)))).reshape(-1, 1)

        n_layers = len(self.layers)
        x_for_output = self._get_layer_output(n_layers - 1)
        grads['output_W'] = x_for_output.T @ d_out
        grads['output_b'] = np.sum(d_out, axis=0)
        dx = d_out @ self.output_W.T

        for i in range(n_layers - 1, -1, -1):
            proj = self.residual_projections[i]
            d_residual = dx.copy()

            mask = self.cache['dropout_masks'][i]
            if mask is not None:
                dx = dx * mask
            dx = dx * (self.cache['post_bn'][i] > 0).astype(np.float64)

            d_bn_gamma, d_bn_beta, dz = self._bn_backward(dx, i)
            grads[f'layer{i}_bn_gamma'] = d_bn_gamma
            grads[f'layer{i}_bn_beta'] = d_bn_beta

            identity = self.cache['residual_identity'][i]
            grads[f'layer{i}_W'] = identity.T @ dz
            grads[f'layer{i}_b'] = np.sum(dz, axis=0)
            dx_linear = dz @ self.layers[i]['W'].T

            if proj is not None:
                grads[f'res_proj_{i}_W'] = identity.T @ d_residual
                grads[f'res_proj_{i}_b'] = np.sum(d_residual, axis=0)
                dx = dx_linear + d_residual @ proj['W'].T
            else:
                dx = dx_linear + d_residual

        return grads, dx

    def _build_param_updates(self, grads, d_emb, variety_ids):
        param_updates = {}
        param_updates['output_W'] = (self.output_W, grads['output_W'])
        param_updates['output_b'] = (self.output_b, grads['output_b'])
        n_layers = len(self.layers)
        for i in range(n_layers):
            param_updates[f'layer{i}_W'] = (self.layers[i]['W'], grads[f'layer{i}_W'])
            param_updates[f'layer{i}_b'] = (self.layers[i]['b'], grads[f'layer{i}_b'])
            param_updates[f'layer{i}_bn_gamma'] = (self.layers[i]['bn_gamma'], grads[f'layer{i}_bn_gamma'])
            param_updates[f'layer{i}_bn_beta'] = (self.layers[i]['bn_beta'], grads[f'layer{i}_bn_beta'])
        for i, proj in enumerate(self.residual_projections):
            if proj is not None:
                param_updates[f'res_proj_{i}_W'] = (proj['W'], grads[f'res_proj_{i}_W'])
                param_updates[f'res_proj_{i}_b'] = (proj['b'], grads[f'res_proj_{i}_b'])
        emb_grad_accum = {}
        for j, vid in enumerate(variety_ids):
            if vid not in emb_grad_accum:
                emb_grad_accum[vid] = []
            emb_grad_accum[vid].append(d_emb[j])
        for vid, grad_list in emb_grad_accum.items():
            avg_grad = np.mean(grad_list, axis=0)
            param_updates[f'emb_{vid}'] = (self.embedding[vid], avg_grad)
        return param_updates

    def backward(self, scores, month_feat, variety_ids, targets, lr=None):
        """反向传播 + Adam更新"""
        preds = self.forward(scores, month_feat, variety_ids)
        self.cache['_preds'] = preds
        hybrid_loss, _, _ = self.compute_loss(preds, targets)

        d_pred = 2.0 * (preds - targets) / len(targets) * MSE_WEIGHT

        if BCE_WEIGHT > 0:
            soft_t = soft_label(targets)
            eps = 1e-7
            preds_c = np.clip(preds, eps, 1 - eps)
            d_bce = (-soft_t / preds_c + (1 - soft_t) / (1 - preds_c)) / len(targets) * BCE_WEIGHT
            mse_grad_norm = np.max(np.abs(d_pred)) if np.max(np.abs(d_pred)) > 0 else 1.0
            bce_grad_norm = np.max(np.abs(d_bce)) if np.max(np.abs(d_bce)) > 0 else 1.0
            if bce_grad_norm > 5.0 * mse_grad_norm:
                scale_factor = 5.0 * mse_grad_norm / bce_grad_norm
                d_bce *= scale_factor
            d_pred = d_pred + d_bce

        grads, dx = self._compute_grads(d_pred)
        emb_start, emb_end = self.cache['_emb_slice']
        d_emb = dx[:, emb_start:emb_end]
        param_updates = self._build_param_updates(grads, d_emb, variety_ids)
        self.optimizer.step(param_updates)
        return hybrid_loss

    def get_state(self):
        state = {
            'embedding': self.embedding.copy(),
            'layers': [{k: v.copy() if isinstance(v, np.ndarray) else v for k, v in layer.items()}
                       for layer in self.layers],
            'output_W': self.output_W.copy(),
            'output_b': self.output_b.copy(),
            'residual_projections': [
                {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in proj.items()}
                if proj is not None else None
                for proj in self.residual_projections
            ],
        }
        return state

    def set_state(self, state):
        self.embedding = state['embedding'].copy()
        for i, layer_state in enumerate(state['layers']):
            for k, v in layer_state.items():
                self.layers[i][k] = v.copy() if isinstance(v, np.ndarray) else v
        self.output_W = state['output_W'].copy()
        self.output_b = state['output_b'].copy()
        if 'residual_projections' in state:
            for i, proj_state in enumerate(state['residual_projections']):
                if proj_state is not None and self.residual_projections[i] is not None:
                    for k, v in proj_state.items():
                        self.residual_projections[i][k] = v.copy() if isinstance(v, np.ndarray) else v

    def save_weights(self, path):
        """保存模型权重（npz格式，v4.0键名）"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        sd = {}
        sd['embedding'] = self.embedding
        for i, layer in enumerate(self.layers):
            sd[f'fc{i}.W'] = layer['W']
            sd[f'fc{i}.b'] = layer['b']
            sd[f'bn{i}.gamma'] = layer['bn_gamma']
            sd[f'bn{i}.beta'] = layer['bn_beta']
            sd[f'bn{i}.running_mean'] = layer['bn_running_mean']
            sd[f'bn{i}.running_var'] = layer['bn_running_var']
        proj_idx = 0
        for i, proj in enumerate(self.residual_projections):
            if proj is not None:
                sd[f'res_proj.{proj_idx}.W'] = proj['W']
                sd[f'res_proj.{proj_idx}.b'] = proj['b']
                proj_idx += 1
        sd['output.W'] = self.output_W
        sd['output.b'] = self.output_b
        sd['_hidden_dims'] = np.array(self.hidden_dims)
        sd['_input_scores'] = np.array(self.input_scores)  # 记录输入评分维度数
        sd['_version'] = np.array([4, 0])
        np.savez(path, **sd)
        print(f'模型权重已保存至 {path}')

    def load_weights(self, path):
        """加载模型权重（v4.2+：48维输入）"""
        data = dict(np.load(path, allow_pickle=True))

        # 检查架构
        loaded_hidden = data.get('_hidden_dims', np.array([24, 16, 8]))
        loaded_input_scores = data.get('_input_scores', np.array([14]))

        loaded_hidden_list = loaded_hidden.tolist()
        loaded_input_scores_val = int(loaded_input_scores.tolist()[0]) if len(loaded_input_scores.shape) > 0 else int(loaded_input_scores)

        if loaded_hidden_list != self.hidden_dims or loaded_input_scores_val != self.input_scores:
            print(f'⚠️ 权重架构不匹配: 文件 hidden={loaded_hidden_list}, input_scores={loaded_input_scores_val} '
                  f'vs 模型 hidden={self.hidden_dims}, input_scores={self.input_scores}')
            raise ValueError(
                f'权重架构不匹配，请重新运行预训练: python pretrain_numpy.py --action pretrain'
            )

        self.embedding = data['embedding'].copy()

        # P0修复: 校验embedding行数（品种扩展后防止旧权重静默覆盖）
        if self.embedding.shape[0] != self.num_varieties:
            raise ValueError(
                f'Embedding行数不匹配: 权重文件={self.embedding.shape[0]}, 模型={self.num_varieties}品种。'
                f'请重新运行预训练: python pretrain_numpy.py --action pretrain'
            )

        is_new_format = 'fc0.W' in data

        # 字段缺失检查
        layer_key_templates_new = ['fc{}.W', 'fc{}.b', 'bn{}.gamma', 'bn{}.beta',
                                   'bn{}.running_mean', 'bn{}.running_var']
        missing_keys = []
        for i in range(len(self.layers)):
            for t in layer_key_templates_new:
                k_i = t.format(i)
                if k_i not in data:
                    missing_keys.append(k_i)
        for k in ['output.W', 'output.b']:
            if k not in data:
                missing_keys.append(k)
        if missing_keys:
            raise KeyError(f'权重文件 {path} 缺少字段: {missing_keys}')

        for i in range(len(self.layers)):
            self.layers[i]['W'] = data[f'fc{i}.W'].copy()
            self.layers[i]['b'] = data[f'fc{i}.b'].copy()
            self.layers[i]['bn_gamma'] = data[f'bn{i}.gamma'].copy()
            self.layers[i]['bn_beta'] = data[f'bn{i}.beta'].copy()
            self.layers[i]['bn_running_mean'] = data[f'bn{i}.running_mean'].copy()
            self.layers[i]['bn_running_var'] = data[f'bn{i}.running_var'].copy()

        proj_idx = 0
        for i, proj in enumerate(self.residual_projections):
            if proj is not None:
                kw, kb = f'res_proj.{proj_idx}.W', f'res_proj.{proj_idx}.b'
                if kw in data:
                    proj['W'] = data[kw].copy()
                    proj['b'] = data[kb].copy()
                proj_idx += 1

        self.output_W = data['output.W'].copy()
        self.output_b = data['output.b'].copy()

        version = data.get('_version', np.array([0, 0]))
        v_tuple = version.tolist() if version.ndim > 0 else [int(version), 0]
        print(f'模型权重已从 {path} 加载 [v{v_tuple[0]}.{v_tuple[1]}, '
              f'hidden={loaded_hidden_list}, input_scores={loaded_input_scores_val}]')


# ============================================================
# 数据加载
# ============================================================
def load_csv_samples(path, filter_invalid=True):
    """加载CSV样本，格式：dim1..dim38（CA统一38维评分）

    Args:
        path: CSV文件路径
        filter_invalid: 是否过滤无效样本（默认True）
            - dim值<0 的样本：CA评分未填充，排除
            - raw_change=0 的样本：y值未回填真实涨跌，排除
    """
    samples = []
    if not os.path.exists(path):
        return samples
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # 读取38维CA评分（dim1-dim38）
                scores = [float(row[f'dim{i}']) for i in range(1, INPUT_TOTAL_SCORES + 1)]

                # 过滤无效样本
                if filter_invalid:
                    # CA评分未填充（dim值为-1占位）
                    if any(s < 0 for s in scores):
                        continue
                    # y值未回填（raw_change=0 表示次日涨跌未知）
                    raw_change = row.get('raw_change', '0.0')
                    if raw_change is None or raw_change == '':
                        raw_change = '0.0'
                    if abs(float(raw_change)) < 1e-8:
                        continue

                samples.append({
                    'scores': scores,  # 38维
                    'month': int(row['month']),
                    'variety_id': int(row['variety_id']),
                    'y': float(row['y']),
                    'date': row.get('date', ''),
                })
            except (ValueError, KeyError):
                continue
    return samples


def encode_batch(samples, indices):
    """将一批样本编码为numpy数组（38维评分）"""
    scores_list, month_feats, variety_ids, ys = [], [], [], []
    for idx in indices:
        s = samples[idx]
        scores_list.append(s['scores'])  # 38维
        m = s['month']
        month_feats.append([math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12)])
        variety_ids.append(s['variety_id'])
        ys.append(s['y'])
    return (np.array(scores_list, dtype=np.float64),
            np.array(month_feats, dtype=np.float64),
            np.array(variety_ids, dtype=np.int32),
            np.array(ys, dtype=np.float64))


# ============================================================
# 预训练
# ============================================================

def load_pretrain_samples(npz_path):
    """加载合成预训练样本

    Args:
        npz_path: pretrain_samples.npz 路径
    Returns:
        list[dict]: samples格式同load_csv_samples
    """
    if not os.path.exists(npz_path):
        print(f"预训练样本不存在: {npz_path}，请先运行 generate_pretrain.py")
        return []

    data = np.load(npz_path)
    X = data['X']
    y = data['y']
    n = len(y)
    print(f"加载预训练样本: {n}个, X.shape={X.shape}")

    samples = []
    for i in range(n):
        samples.append({
            'scores': X[i].tolist(),
            'month': random.randint(1, 12),
            'variety_id': random.randint(0, NUM_VARIETIES - 1),
            'y': float(y[i]),
            'date': f'syn_{i:05d}',
        })
    return samples


def pretrain_numpy(
    model, samples,
    epochs=50, lr=0.001, batch_size=64, verbose=True,
):
    """NumPy预训练 v4.3: 合成数据初始化模型权重

    相比微调的区别：
    - 学习率更大 (0.001 vs 1e-4)，加速收敛
    - 不做时序划分（合成数据无时间顺序）
    - 不设早停 (大量合成样本无需担心过拟合)
    - 纯MSE损失 (BCE=0)
    """
    n = len(samples)
    if n < 100:
        if verbose:
            print(f"预训练样本不足100（当前{n}），跳过")
        return model

    steps_per_epoch = max(1, n // batch_size)
    warmup_steps = 3 * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    model.optimizer = AdamOptimizer(lr=lr, warmup_steps=warmup_steps, total_steps=total_steps)
    for vid in range(model.num_varieties):
        model.optimizer.set_lr_scale(f'emb_{vid}', 5.0)

    for epoch in range(1, epochs + 1):
        model.training = True
        epoch_loss = 0.0
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            b_scores, b_month, b_vids, b_ys = encode_batch(samples, batch_idx)
            loss = model.backward(b_scores, b_month, b_vids, b_ys)
            epoch_loss += loss * len(batch_idx)
        epoch_loss /= n

        if verbose and epoch % 10 == 0:
            print(f'[Pretrain] Epoch {epoch:3d}/{epochs}: loss={epoch_loss:.6f}, lr={model.optimizer.lr:.6f}')

    if verbose:
        print(f'[Pretrain] 完成: final_loss={epoch_loss:.6f}')
    return model


# ============================================================
# 微调主流程
# ============================================================
def finetune_numpy(
    model, samples,
    epochs=100, lr=1e-4, batch_size=32, patience=15,
    min_delta=1e-4, val_ratio=0.2, verbose=True,
):
    """NumPy微调v4.0：真实数据，小学习率，分层LR，时序划分"""
    n = len(samples)
    if n < 30:
        if verbose:
            print(f"样本数不足30（当前{n}），跳过微调")
        return model, [], float('inf')

    sorted_samples = sorted(samples, key=lambda s: s.get('date', ''))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val
    train_samples = sorted_samples[:n_train]
    val_samples = sorted_samples[n_train:]
    if verbose:
        print(f"时序划分: train={n_train}(最早), val={n_val}(最新)")

    val_scores, val_month, val_vids, val_ys = encode_batch(val_samples, range(n_val))

    steps_per_epoch = max(1, n_train // batch_size)
    warmup_steps = 5 * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    model.optimizer = AdamOptimizer(lr=lr, warmup_steps=warmup_steps, total_steps=total_steps)
    for vid in range(model.num_varieties):
        model.optimizer.set_lr_scale(f'emb_{vid}', 5.0)

    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    log = []

    for epoch in range(1, epochs + 1):
        model.training = True
        epoch_loss = 0.0
        indices = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            batch_idx = indices[start:start + batch_size]
            b_scores, b_month, b_vids, b_ys = encode_batch(train_samples, batch_idx)
            loss = model.backward(b_scores, b_month, b_vids, b_ys)
            epoch_loss += loss * len(batch_idx)
        epoch_loss /= n_train

        model.training = False
        val_preds = model.forward(val_scores, val_month, val_vids)
        val_loss, _, _ = model.compute_loss(val_preds, val_ys)
        log.append({'epoch': epoch, 'train_loss': round(epoch_loss, 6), 'val_loss': round(val_loss, 6)})

        if verbose and epoch % 10 == 0:
            print(f'[Finetune] Epoch {epoch}: train={epoch_loss:.6f}, val={val_loss:.6f}')

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = model.get_state()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f'[Finetune] Early stopping at epoch {epoch}, best_val={best_val_loss:.6f}')
                break

    if best_state is not None:
        model.set_state(best_state)
    return model, log, best_val_loss


# ============================================================
# 推理
# ============================================================
def predict_numpy(model, scores_list, months, variety_ids):
    """批量推理，返回 list[float] CANN评分

    Args:
        model: NumpyCANNModel实例
        scores_list: 38维CA评分 list[list[float]]
        months: 月份列表 list[int]
        variety_ids: 品种ID列表 list[int]
    """
    model.training = False
    scores_arr = np.array(scores_list, dtype=np.float64)
    month_feats = np.array(
        [[math.sin(2 * math.pi * m / 12), math.cos(2 * math.pi * m / 12)] for m in months],
        dtype=np.float64)
    vid_arr = np.array(variety_ids, dtype=np.int32)
    preds = model.forward(scores_arr, month_feats, vid_arr)
    return [round(float(p), 4) for p in preds]


def load_pretrained_model(data_dir=None):
    """加载预训练NumPy模型，供每日任务调用"""
    if data_dir is None:
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(_script_dir, '..', '..', '..', 'CANN', 'data')
        data_dir = os.path.normpath(data_dir)
    candidates = [
        os.path.join(data_dir, "model_weights.npz"),
        os.path.join(data_dir, "model_weights_pretrained.npz"),
    ]
    for wf in candidates:
        if os.path.exists(wf):
            try:
                model = NumpyCANNModel()
                model.load_weights(wf)
                return model, wf
            except Exception as e:
                print(f"加载权重 {wf} 失败: {e}")
                continue
    return None, None


def predict_single(model, dim_scores, month, variety_id):
    """单品种推理便捷函数（含异常边界处理）。
    Args:
        model: NumpyCANNModel实例（已加载权重），None时返回0.5
        dim_scores: 38维CA评分 list[float]，None/空时返回0.5
        month: 月份 1-12
        variety_id: 品种ID 0-52，越界自动裁剪
    Returns:
        float: CANN评分 [0,1]
    """
    # v4.2: 异常边界处理
    if model is None:
        warnings.warn("CANN模型未加载，返回中性评分0.5")
        return 0.5
    if dim_scores is None or (isinstance(dim_scores, (list, tuple)) and len(dim_scores) == 0):
        warnings.warn("CA评分为None或空，返回中性评分0.5")
        return 0.5
    # 品种ID边界检查
    max_vid = model.num_varieties - 1
    if variety_id < 0:
        warnings.warn(f"品种ID {variety_id} 非法，修正为0")
        variety_id = 0
    elif variety_id > max_vid:
        warnings.warn(f"品种ID {variety_id} 超出范围({max_vid})，修正为{max_vid}")
        variety_id = max_vid

    results = predict_numpy(model, [dim_scores], [month], [variety_id])
    return results[0] if results else 0.5


# ============================================================
# 每日训练主流程
# ============================================================
def run_daily_training_numpy(data_dir, verbose=True, use_pretrained=True):
    """每日训练流程（纯NumPy版v4.3）

    v4.3: 优先检查预训练权重 ->
      无预训练权重时自动用合成样本预训练 ->
      加载预训练权重微调 ->
      保存最终权重
    """
    csv_path = os.path.join(data_dir, "historical_samples.csv")
    model_path = os.path.join(data_dir, "model_weights.npz")
    pretrained_path = os.path.join(data_dir, "model_weights_pretrained.npz")
    pretrain_npz = os.path.join(data_dir, "pretrain_samples.npz")

    def _find_weights():
        for p in [model_path, pretrained_path]:
            if os.path.exists(p):
                return p
        return None

    # ---- Phase 1: 预训练（如需要） ----
    model = NumpyCANNModel()
    has_pretrained = os.path.exists(pretrained_path)

    if not has_pretrained and use_pretrained and os.path.exists(pretrain_npz):
        if verbose:
            print("无预训练权重，开始合成数据预训练...")
        pt_samples = load_pretrain_samples(pretrain_npz)
        if pt_samples:
            model = pretrain_numpy(model, pt_samples, verbose=verbose)
            model.save_weights(pretrained_path)
            has_pretrained = True
            if verbose:
                print(f"预训练权重已保存: {pretrained_path}")
    elif not has_pretrained and not os.path.exists(pretrain_npz):
        if verbose:
            print(f"无预训练样本 ({pretrain_npz})，跳过预训练（随机初始化）")

    # ---- Phase 2: 微调 ----
    samples = load_csv_samples(csv_path)
    if verbose:
        print(f"加载历史样本数: {len(samples)}")

    if len(samples) < 30:
        if verbose:
            print(f"样本数不足30（当前{len(samples)}），跳过微调")
        if has_pretrained:
            model = NumpyCANNModel()
            model.load_weights(pretrained_path)
            if verbose:
                print(f"已加载预训练权重（无微调）: {pretrained_path}")
        return model

    weight_file = _find_weights()
    if weight_file:
        try:
            model.load_weights(weight_file)
            model.optimizer = AdamOptimizer(lr=0.001)
            for vid in range(model.num_varieties):
                model.optimizer.set_lr_scale(f'emb_{vid}', 5.0)
            if verbose:
                print(f"已加载权重({weight_file})，微调模式")
            model, log, best_val_loss = finetune_numpy(model, samples, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"加载权重失败: {e}")
            # v4.3: 回退前先检查是否有预训练权重可用
            if os.path.exists(pretrained_path):
                model = NumpyCANNModel()
                model.load_weights(pretrained_path)
                if verbose:
                    print(f"回退到预训练权重: {pretrained_path}")
            else:
                model = NumpyCANNModel()
            model, log, best_val_loss = finetune_numpy(model, samples, verbose=verbose)
    else:
        if verbose:
            print("无可用权重")
        # v4.3: 回退前检查预训练权重
        if os.path.exists(pretrained_path):
            model = NumpyCANNModel()
            model.load_weights(pretrained_path)
            if verbose:
                print(f"回退到预训练权重: {pretrained_path}")
        model, log, best_val_loss = finetune_numpy(model, samples, verbose=verbose)

    model.save_weights(model_path)
    if verbose:
        print(f"模型已保存至 {model_path}, best_val_loss={best_val_loss:.6f}")

    log_path = os.path.join(data_dir, "training_log.csv")
    os.makedirs(data_dir, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "epoch", "train_loss", "val_loss", "num_samples"])
            if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
                writer.writeheader()
            for entry in log:
                writer.writerow({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": entry["epoch"],
                    "train_loss": entry["train_loss"],
                    "val_loss": entry["val_loss"],
                    "num_samples": len(samples),
                })
    except Exception as e:
        warnings.warn(f"保存训练日志失败: {e}")

    return model


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CANN NumPy v4.3")
    parser.add_argument("--data-dir", default="CANN/data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--action", default="daily",
                        choices=["pretrain", "finetune", "daily"])
    args = parser.parse_args()

    if args.action == "pretrain":
        # 仅预训练：合成数据 → model_weights_pretrained.npz
        model = NumpyCANNModel(seed=args.seed)
        pretrain_npz = os.path.join(args.data_dir, "pretrain_samples.npz")
        pretrained_path = os.path.join(args.data_dir, "model_weights_pretrained.npz")
        if not os.path.exists(pretrain_npz):
            print(f"错误: 预训练样本不存在 {pretrain_npz}，请先运行 generate_pretrain.py")
            sys.exit(1)
        pt_samples = load_pretrain_samples(pretrain_npz)
        model = pretrain_numpy(model, pt_samples, epochs=args.pretrain_epochs)
        model.save_weights(pretrained_path)
    elif args.action == "finetune":
        # v4.3: finetune前也确保有预训练权重
        model = NumpyCANNModel(seed=args.seed)
        pretrained_path = os.path.join(args.data_dir, "model_weights_pretrained.npz")
        pretrain_npz = os.path.join(args.data_dir, "pretrain_samples.npz")
        if not os.path.exists(pretrained_path) and os.path.exists(pretrain_npz):
            print("无预训练权重，先执行预训练...")
            pt_samples = load_pretrain_samples(pretrain_npz)
            model = pretrain_numpy(model, pt_samples, epochs=args.pretrain_epochs)
            model.save_weights(pretrained_path)
        model_path = os.path.join(args.data_dir, "model_weights.npz")
        weight_file = pretrained_path if os.path.exists(pretrained_path) else None
        if weight_file:
            model.load_weights(weight_file)
        else:
            weight_file = model_path
            if os.path.exists(weight_file):
                model.load_weights(weight_file)
        samples = load_csv_samples(os.path.join(args.data_dir, "historical_samples.csv"))
        model, log, bvl = finetune_numpy(model, samples, epochs=args.epochs)
        model.save_weights(model_path)
    elif args.action == "daily":
        run_daily_training_numpy(args.data_dir)
