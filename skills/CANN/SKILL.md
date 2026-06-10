# CANN (CryptoAnalysis Neural Network) — 加密货币神经网络评分技能

## 描述
基于 CA 技能14维度基本面评分 + 24维度技术面评分，通过纯NumPy神经网络输出综合多空评分。网络通过真实CA评分与实际涨跌的对应关系进行监督学习，逐步优化评分精度。

## 简称
CANN

## 触发场景
- 用户说"CANN"、"CANN分析"、"神经网络评分"等
- 日程触发（标题含"CANN"或"CA-NN"）
- 每日 UTC 00:15 自动执行（24/7市场）
- 每4小时增量推理（配合CatTrader调度）

## 依赖
- CryptoAnalysis (CA) 技能：提供14维度基本面评分（**必须来自真实分析**）
- 技术面模块：`skills/CryptoAnalysis/scripts/binance_data.py` 提供24维度技术面评分
- NumPy：纯NumPy实现，无PyTorch/深度学习框架依赖
- Binance API：K线数据和技术面数据

## ⚠️ 铁律：禁止合成数据

- D1-D14（基本面评分）**必须**来自CA技能的真实14维度分析输出
- T1-T24（技术面评分）**必须**来自Binance API真实K线计算
- y值**必须**来自Binance API真实次日涨跌（sigmoid映射）
- **禁止**用任何公式推算/合成y值或其他维度
- 有多少天真实CA数据，就有多少天有效训练样本

---

## 神经网络架构 v4.0（保持与商品期货版一致）

### 输入（48维）

| 序号 | 输入 | 类型 | 编码方式 | 编码后维度 |
|------|------|------|----------|-----------|
| 1-14 | CA基本面评分 | float [0,1] | 直接输入 | 14 |
| 15-38 | 技术面评分 | float [0,1] | 直接输入 | 24 |
| 39-40 | 月份 | int [1,12] | 周期编码 sin/cos | 2 |
| 41-48 | 币种ID | int [0,49] | Embedding(dim=8) | 8 |

**月份周期编码**：`month_sin = sin(2π × month / 12)`，`month_cos = cos(2π × month / 12)`。加密市场也有月度季节性（如"Uptober"、年末rally等）。

**币种ID嵌入**：通过 Embedding 层学习每个币种的潜在特征表示，维度为8。

**有效输入总维度**：14 + 24 + 2 + 8 = **48**

### 网络结构（不变）

```
Input: scores(14基本面+24技术面=38) + month_sin + month_cos + crypto_embedding(8) = 48
  │
  ├─ Linear(48, 48) → BatchNorm(48) → ReLU → Dropout(0.25) + 残差连接
  │
  ├─ Linear(48, 32) → BatchNorm(32) → ReLU → Dropout(0.25) + 残差投影(48→32)
  │
  ├─ Linear(32, 16) → BatchNorm(16) → ReLU → Dropout(0.25) + 残差投影(32→16)
  │
  ├─ Linear(16, 8)  → BatchNorm(8)  → ReLU → Dropout(0.25) + 残差投影(16→8)
  │
  └─ Linear(8, 1) → Sigmoid
       │
       └─ Output: y ∈ [0, 1]
```

### 训练超参数（不变）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Loss | MSE | 回归任务，输出连续值[0,1] |
| Optimizer | Adam（手写NumPy） | 自适应学习率，支持分层LR |
| Learning Rate | 5e-5 | 微调学习率（embedding层lr×5.0） |
| Batch Size | 32 | 微调批量 |
| Epochs | 100 | 最大训练轮数 |
| Early Stopping Patience | 15 | 验证集连续15轮不改善则停止 |
| Train/Val Split | 80/20 | 时序划分（前80%训练，后20%验证） |
| Gradient Clip Norm | 1.0 | 全局梯度裁剪 |

---

## 每日执行流程

### 第零步：币种元数据验证
确保 Top50 列表与Binance实际可交易合约一致，检测下架或新上架合约。

### 第一步：采集技术面（自动）
通过 Binance API 获取50币种K线数据，计算24维技术面评分：
```python
from skills.CryptoAnalysis.scripts.binance_data import BinanceDataProvider
from skills.common.technical_indicators import compute_from_klines

with BinanceDataProvider() as provider:
    for vid in range(50):
        klines = provider.get_klines(VARIETY_CODES[vid], interval="1d", limit=200)
        tech_scores = compute_from_klines(klines)  # 返回24维[0,1]评分
```

技术面数据写入 `daily_scores/scores_YYYYMMDD.csv`，D1-D14列暂填-1（待CA填充）。

### 第二步：CA评分采集（必须手动执行或半自动）
逐币种调用CA技能获取真实14维评分，然后写入CSV：
```python
from skills.CANN.scripts.daily_pipeline import write_ca_scores
write_ca_scores('YYYYMMDD', crypto_id, [d1,...,d14], './CANN/data/daily_scores')
# 输入校验：币种ID 0-49，CA评分14维且每维在[0,1]
```

⚠️ CA评分必须来自真实分析，禁止合成/生成。50币种约需30-60分钟。

### 第三步：计算真实涨跌标签 y 值
使用 Binance API 获取次日真实涨跌幅（UTC 00:00为日切）：
```python
ret = (next_close - today_close) / today_close
y = 1 / (1 + exp(-ret * 20))  # sigmoid映射到[0,1]
```

| 涨跌幅 ret | y值 | 含义 |
|------------|------|------|
| +5% | 0.97 | 强利多 |
| +2% | 0.88 | 偏多 |
| 0% | 0.50 | 中性 |
| -2% | 0.12 | 偏空 |
| -5% | 0.03 | 强利空 |

**注意**：加密波动率高于商品期货，系数20保持合理映射范围。对于极端行情（±10%+），y值接近饱和。

**y值占位规则**：当日样本的y值用0.5占位（raw_change=0标记），次日自动回填真实涨跌。**禁止用公式推算y值。**

### 第四步：训练神经网络
仅使用**同时具备真实CA+真实y值**的有效样本训练。

**训练策略**：
- 从 `CANN/data/historical_samples.csv` 加载有效历史数据
- 将当日新样本追加到历史数据
- 全量有效数据微调网络
- 使用早停法防止过拟合
- 保存模型权重到 `CANN/data/model_weights.npz`

**冷启动**：有效样本<30条时不训练，推理结果为0.5（中性）。

### 第五步：推理评分
用训练好的网络对50币种计算综合评分：
```python
from skills.CANN.scripts.pretrain_numpy import predict_single
cann_score = predict_single(model, dim_scores_14d, month, crypto_id, tech_scores=tech_scores_24d)
```

- 输入：14维CA评分 + 24维技术面评分 + 当前月份 + 币种ID
- 输出：综合评分 y ∈ [0,1]
- 模型未训练时返回0.5

**多空研判对照表**（CANN评分）：

| 区间 | 判定 | 颜色标记 |
|------|------|----------|
| 0.00-0.35 | 强利空 | 🔴 |
| 0.35-0.45 | 偏空 | 🟠 |
| 0.45-0.55 | 中性 | 🟡 |
| 0.55-0.65 | 偏多 | 🟢 |
| 0.65-1.00 | 强利多 | 🔵 |

### 第六步：生成报告
在一张总表中列出所有币种的：
- 14维CA评分
- 24维技术面评分
- CA加权综合评分
- CANN综合评分
- 评分差异（CANN - CA）
- 多空研判
- 资金费率（附加信息）

报告保存到 `CANN/reports/CANN报告_YYYYMMDD.md`

---

## 数据存储结构

```
./CANN/
├── data/
│   ├── crypto_meta.json              # Top50币种元数据
│   ├── historical_samples.csv        # 累积训练数据（仅含有效样本）
│   ├── daily_scores/                 # 每日评分
│   │   ├── scores_20260610.csv       # 格式：date,crypto_id,name,month,dim1..14,tech1..24
│   │   └── cann_results_20260610.json
│   ├── model_weights.npz             # 当前模型权重
│   ├── model_weights_pretrained.npz  # 预训练权重
│   └── training_log.csv              # 训练日志
├── references/
│   └── 币种列表.md                   # Top50 USDT永续映射
├── scripts/
│   ├── pretrain_numpy.py             # 核心：模型定义+训练+推理
│   ├── daily_pipeline.py             # 每日管线：技术面→CA写入→回填→微调→推理
│   └── ca_scorer.py                  # CA评分引擎
└── SKILL.md
```

### CSV格式示例（daily_scores/scores_YYYYMMDD.csv）
```
date,crypto_id,crypto_name,month,dim1,dim2,...,dim14,tech1,tech2,...,tech24
2026-06-10,0,BTC,6,0.6,0.5,0.65,0.55,0.45,0.6,0.4,0.55,0.5,0.55,0.5,0.45,0.6,0.72,0.35,...
```

### historical_samples.csv 额外列
```
...,y,raw_change
...,0.880797,0.020000    ← 有效样本（raw_change≠0）
...,0.500000,0.000000    ← 待回填（raw_change=0，训练时自动过滤）
```

---

## 交易日规则（加密版）

加密市场24/7/365交易，与传统市场不同：
1. **不区分交易日与非交易日**：每天都是交易日
2. **日切时间**：UTC 00:00 作为日切点（对应Binance日K线收盘）
3. **每日管线时间**：UTC 00:15（日切后15分钟，数据已稳定）
4. **增量推理**：每4小时执行一次轻量推理（配合CatTrader），不重新训练
5. **训练频率**：每日一次（跟随每日管线）

---

## 冷启动策略

由于是新系统，初始时无历史CA数据：
1. **阶段一（0-30个有效样本）**：不训练，CANN输出=0.5（中性）
   - CatTrader全部s=0.5，不产生任何开仓信号
2. **阶段二（30-100个有效样本）**：开始训练，但使用较高Dropout(0.35)防过拟合
   - CANN开始输出有意义的方向信号
3. **阶段三（>100个有效样本）**：正常训练，标准超参数
   - 模型趋于稳定

建议在阶段一期间，手动执行CA分析积累训练数据，每天至少覆盖10-15个核心币种。

---

## 注意事项

1. **冷启动**：训练数据不足30个有效样本时不训练，CANN评分为0.5（中性）
2. **无权重不推理**：无模型权重时不使用随机初始化模型推理，统一返回0.5
3. **CA评分质量**：CANN精度直接依赖CA评分质量，确保CA按规范执行
4. **过拟合防护**：数据量较小时，Dropout和早停法是关键防线
5. **纯NumPy**：不依赖PyTorch或任何深度学习框架
6. **加密波动率**：加密市场波动大，y值sigmoid映射保留系数20避免过度饱和
7. **24/7市场**：没有收盘概念，日切点UTC 00:00视为"收盘"
8. **品种数量**：50个Binance USDT永续合约品种
9. **免责声明**：所有报告必须附带免责声明，本分析仅供参考，不构成投资建议
