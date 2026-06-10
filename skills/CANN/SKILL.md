# CANN (StockAnalysis Neural Network) — 美股TradFi永续神经网络评分技能

## 描述
基于 SA 技能14维度基本面评分 + 24维度技术面评分，通过纯NumPy神经网络输出综合多空评分。网络通过真实SA评分与实际涨跌的对应关系进行监督学习，逐步优化评分精度。

## 简称
CANN（与商品期货/加密版共用名称和架构）

## 触发场景
- 用户说"CANN"、"CANN分析"、"神经网络评分"等
- 日程触发（标题含"CANN"）
- 每日 UTC 00:15 自动执行

## 依赖
- StockAnalysis (SA) 技能：提供14维度基本面评分（**必须来自真实分析**）
- 技术面模块：`skills/StockAnalysis/scripts/binance_data.py` 提供24维度技术面评分
- NumPy：纯NumPy实现，无PyTorch/深度学习框架依赖

## ⚠️ 铁律：禁止合成数据

- D1-D14（基本面评分）**必须**来自SA技能的真实14维度分析输出
- T1-T24（技术面评分）**必须**来自Binance API真实K线计算
- y值**必须**来自Binance API真实次日涨跌（sigmoid映射）
- **禁止**用任何公式推算/合成y值或其他维度
- 有多少天真实SA数据，就有多少天有效训练样本

---

## 神经网络架构 v4.0

### 输入（48维）

| 序号 | 输入 | 类型 | 编码方式 | 编码后维度 |
|------|------|------|----------|-----------|
| 1-14 | SA基本面评分 | float [0,1] | 直接输入 | 14 |
| 15-38 | 技术面评分 | float [0,1] | 直接输入 | 24 |
| 39-40 | 月份编码 | int [1,12] | 周期编码 sin/cos | 2 |
| 41-48 | 品种嵌入 | Embedding(dim=8) | 可学习查表 | 8 |

**月份周期编码**：`month_sin = sin(2π × month / 12)`，`month_cos = cos(2π × month / 12)`，捕捉月份的周期性（12月与1月相邻，避免线性编码的边界断裂）。

**品种ID嵌入**：通过8维可学习Embedding层，让网络自动学习每个品种的潜在行为特征。相似涨跌模式的品种（如NVDA和AMD同属半导体）在嵌入空间中会自然靠近，无需人工定义类别。

**有效输入总维度**：14 + 24 + 2 + 8 = **48**

### 14维SA基本面评分（D1-D14）

| 序号 | 维度 | 类别 | 数据窗口 |
|------|------|------|----------|
| D1 | 货币政策 | 宏观 | 30天 |
| D2 | 经济周期 | 宏观 | 30天 |
| D3 | 财政政策 | 宏观 | 30天 |
| D4 | 行业景气度 | 行业 | 7天 |
| D5 | 比较优势 | 行业 | 30天 |
| D6 | 盈利能力 | 公司 | 30天 |
| D7 | 估值安全边际 | 公司 | 24小时 |
| D8 | 盈利预期 | 公司 | 30天 |
| D9 | 合约资金流 | 市场 | 4小时 |
| D10 | 机构动向 | 市场 | 7天 |
| D11 | 市场情绪 | 市场 | 4小时 |
| D12 | 技术结构 | 市场 | 24小时 |
| D13 | 公司治理 | 治理 | 30天 |
| D14 | 事件驱动 | 催化 | 7天 |

### 24维技术面评分（T1-T24）

由 `binance_data.py` 的 `get_technical_indicators()` 从Binance K线自动计算：

| 序号 | 指标 | 说明 |
|------|------|------|
| T1 | 价格位置 | (close - MA20) / MA20 → [0,1] |
| T2 | MA20偏离 | (close - MA50) / MA50 → [0,1] |
| T3 | MA50偏离 | (close - MA200) / MA200 → [0,1] |
| T4 | MA排列 | bullish/bearish/mixed → [1.0, 0.5, 0.0] |
| T5 | 布林带上轨偏离 | (close - upper) / (upper - lower) → [0,1] |
| T6 | 布林带下轨偏离 | (close - lower) / (upper - lower) → [0,1] |
| T7 | 布林带宽 | (upper-lower) / middle → 归一化[0,1] |
| T8 | RSI14 | [0,100] → /100 → [0,1] |
| T9 | ATR% | 归一化[0,1] |
| T10-13 | 各周期动量 | 1d/5d/10d/20d 涨跌幅归一化 |
| T14-17 | 成交量 | 1d/5d/10d/20d 量比归一化 |
| T18-21 | 高低位 | 距20日/50日/200日高低位的百分比 |
| T22-24 | 波动率 | 20日/50日/200日历史波动率分位 |

⚠️ **T1-T24全部由Binance API自动计算**，无需人工干预。SA技能只需填充D1-D14。

### 网络结构

```
Input: scores(14基本面+24技术面=38) + month_sin + month_cos + variety_embedding(8) = 48
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

### 架构选择理由

| 设计决策 | 理由 |
|----------|------|
| 4层隐藏层（48→32→16→8）+ 残差连接 | 适度深度，残差连接防止梯度消失、加速收敛 |
| BatchNorm（手写NumPy） | 稳定各层输入分布，缓解小批量训练不稳定 |
| Dropout(0.25) | 适度正则化，防止小数据集过拟合 |
| Sigmoid输出 | 输出范围严格 [0,1]，与SA评分语义一致 |
| Embedding(8) | 35个品种，8维嵌入足以捕捉品种间行为差异 |
| 月份sin/cos编码 | 保留周期性（12月≈1月），避免线性编码破坏边界关系 |
| 纯NumPy实现 | 零外部依赖，沙箱环境兼容 |

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| MSE | 1.0 | 主损失，回归任务 [0,1] |
| 软标签BCE | 0.0（已禁用） | 梯度是MSE的100+倍，导致不收敛 |
| 对比学习 | 0.0（已禁用） | 梯度注入与主损失Adam更新冲突 |

**MSE损失公式**：`Loss = mean((y_pred - y_true)²)`

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Loss | MSE | 回归任务，输出连续值[0,1] |
| Optimizer | Adam（手写NumPy） | 自适应学习率，支持分层LR |
| Learning Rate | 5e-5 | 基础学习率（全连接层） |
| Embedding LR | 2.5e-4 (×5.0) | 嵌入层学习率 ×5（加速品种表征学习） |
| Batch Size | 32 | 全量训练批量 |
| Epochs | 100 | 最大训练轮数 |
| Early Stopping Patience | 15 | 验证集连续15轮loss不改善则停止 |
| Train/Val Split | 80/20 | 时序划分（前80%训练，后20%验证） |
| Gradient Clip Norm | 1.0 | 全局梯度裁剪，防止梯度爆炸 |
| Dropout (训练) | 0.25 | 训练时随机丢弃25%神经元 |
| Dropout (推理) | 0.0 | 推理时关闭Dropout |

---

## 每日执行流程（UTC 00:15 自动触发）

### 第一步：采集技术面（自动）

通过Binance API获取35品种日线K线数据，计算24维技术面评分：

```python
from skills.StockAnalysis.scripts.binance_data import BinanceDataProvider

with BinanceDataProvider() as provider:
    for vid, sym in VARIETY_CODES.items():
        tech = provider.get_technical_indicators(sym, interval="1d", limit=200)
        # tech 返回 dict 含 price, ma20, ma50, ma200, bb_upper, bb_lower, 
        #              atr, atr_pct, rsi14, high_20d, low_20d, 
        #              price_percentile_200d, ma_alignment

# 将原始指标转为24维[0,1]评分写入 daily_scores/scores_YYYYMMDD.csv
# D1-D14列暂填-1（待SA填充）
```

技术面数据自动写入 `daily_scores/scores_YYYYMMDD.csv`，D1-D14列暂填-1（等待SA手动填充）。

### 第二步：SA评分采集（必须手动执行）

逐品种调用SA技能获取真实14维评分，然后写入CSV：

```python
from skills.CANN.scripts.daily_pipeline import write_ca_scores

write_ca_scores('YYYYMMDD', variety_id, [d1, d2, ..., d14], './CANN/data/daily_scores')
# 输入校验：品种ID 0-34，CA评分14维且每维在[0,1]范围内
```

⚠️ **CA评分必须来自SA技能真实分析**，禁止合成/生成。35品种完整分析约需30-60分钟。

### 第三步：计算真实涨跌标签 y 值（次日）

在下一个UTC 00:15管线运行时，用Binance API获取该品种次日真实涨跌：

```python
# 在 daily_pipeline.py 的 update_historical_y() 中执行
ret = (next_day_close - today_close) / today_close
y = 1 / (1 + exp(-ret * 20))  # sigmoid映射到[0,1]
```

| 涨跌幅 ret | y值 | 含义 |
|------------|------|------|
| +10% | 0.88 | 强利多 |
| +5% | 0.73 | 偏多 |
| +3% | 0.65 | 略多 |
| +1% | 0.55 | 微多 |
| 0% | 0.50 | 中性 |
| -1% | 0.45 | 微空 |
| -3% | 0.35 | 略空 |
| -5% | 0.27 | 偏空 |
| -10% | 0.12 | 强利空 |

**y值占位规则**：当日新增样本的y值用0.5占位（raw_change=0标记），次日管线自动回填真实涨跌。**禁止用公式推算y值。**

### 第四步：训练神经网络

仅使用**同时具备真实SA评分 + 真实y值**的有效样本训练：

```python
from skills.CANN.scripts.pretrain_numpy import train_from_csv

# load_csv_samples(filter_invalid=True) 自动过滤：
#   - dim值 < 0（CSV占位-1）
#   - raw_change == 0（y值待回填的占位行）
model = train_from_csv('historical_samples.csv')
```

**训练策略**：
1. 从 `CANN/data/historical_samples.csv` 加载有效历史数据
2. 将当日新样本追加到历史数据（y=0.5占位行自动过滤）
3. 全量有效数据微调网络（小学习率+分层LR+时序划分）
4. 使用早停法（patience=15）防止过拟合
5. 保存模型权重到 `CANN/data/model_weights.npz`
6. 保存训练日志到 `CANN/data/training_log.csv`

**冷启动规则**：

| 有效样本数 | 行为 | 说明 |
|-----------|------|------|
| < 25 | 不训练，全部输出0.5 | 数据不足以学习有意义的模式 |
| 25-100 | Dropout增至0.35训练 | 防过拟合，少量数据需更强正则 |
| > 100 | 标准训练（Dropout 0.25） | 正常模式 |

### 第五步：推理评分

用训练好的网络对35品种计算综合评分：

```python
from skills.CANN.scripts.pretrain_numpy import NumpyCANNModel, predict_single

# 加载模型
model = NumpyCANNModel(num_varieties=35, embedding_dim=8, 
                       input_scores=38, hidden_dims=[48,32,16,8])
model.load_weights('skills/CANN/data/model_weights.npz')

# 单品种推理
cann_score = model.predict_single(
    dim_scores_14d,      # SA 14维 [0,1]
    month,               # 当前月份 1-12
    variety_id,          # 品种ID 0-34
    tech_scores_24d      # 技术面 24维 [0,1]
)
# 输出：float ∈ [0,1]
```

- 无模型权重文件时：统一返回0.5（中性）
- 模型已训练时：正常推理

**多空研判对照表**（CANN评分）：

| 区间 | 判定 | 颜色标记 | CatTrader动作 |
|------|------|----------|--------------|
| 0.00-0.35 | 强利空 | 🔴 | 做空 0.5× |
| 0.35-0.45 | 偏空 | 🟠 | 做空 0.3× |
| 0.45-0.55 | 中性 | 🟡 | 平仓/观望 |
| 0.55-0.65 | 偏多 | 🟢 | 做多 0.3× |
| 0.65-1.00 | 强利多 | 🔵 | 做多 0.5× |

### 第六步：生成报告

在一张总表中列出所有品种的：
- 14维SA评分明细
- 24维技术面概述
- SA加权综合评分
- CANN综合评分
- 评分差异（CANN - SA加权）
- 多空研判及颜色标记
- 品种类别分组汇总

报告保存到 `skills/CANN/reports/CANN报告_YYYYMMDD.md`

---

## 数据存储结构

```
./skills/CANN/
├── SKILL.md
├── data/
│   ├── historical_samples.csv      # 累积训练数据（仅含有效样本：真实SA+真实y）
│   ├── daily_scores/               # 每日评分（技术面自动+SA手动）
│   │   ├── scores_20260610.csv     # 格式：date,variety_id,variety_name,month,dim1..14,tech1..24
│   │   └── ...
│   ├── model_weights.npz           # 当前模型权重（含嵌入矩阵+各层参数+BN统计量）
│   ├── tradfi_meta.json            # 35品种元数据（名称/代码/类别）
│   └── training_log.csv            # 训练日志（日期/样本数/loss/val_loss/epochs）
├── reports/
│   ├── CANN报告_20260610.md
│   └── ...
├── scripts/
│   ├── pretrain_numpy.py           # 核心脚本：模型定义+前向+反向+Adam+训练+推理
│   └── daily_pipeline.py           # 每日管线：技术面采集→CA写入→y值回填→微调→推理
└── references/
    └── 币种列表.md                  # 35品种ID映射+类别分组
```

### CSV格式示例（daily_scores/scores_YYYYMMDD.csv）

```
date,variety_id,variety_name,month,dim1,dim2,...,dim14,tech1,tech2,...,tech24
2026-06-10,0,NVDA,6,0.65,0.55,0.50,0.70,0.80,0.75,0.45,0.60,0.55,0.50,0.45,0.65,0.50,0.55,0.72,0.35,...
2026-06-10,1,AAPL,6,0.55,0.50,0.50,0.65,0.75,0.70,0.55,0.60,0.50,0.45,0.45,0.55,0.50,0.50,0.50,0.48,0.30,...
```

- dim1-dim14：SA基本面评分 [0,1]，**未填充时为-1**（占位值，训练时自动过滤）
- tech1-tech24：技术面评分 [0,1]，**由Binance API自动计算填充**

### historical_samples.csv 额外列

```
...,y,raw_change
...,0.731058,0.010000     ← 有效样本（raw_change≠0，真实y值）
...,0.500000,0.000000     ← 待回填（raw_change=0，训练时 filter_invalid=True 自动过滤）
...,0.269000,-0.012000    ← 有效样本（空头方向）
```

- `y`：sigmoid映射后的真实涨跌标签
- `raw_change`：原始涨跌幅（0=等待回填）
- 训练时 `filter_invalid=True` 过滤 `dim<0` 行和 `raw_change=0` 行

---

## 权重文件格式（v4.0）

模型权重保存为 `.npz` 格式，可通过 `model.save_weights(path)` / `model.load_weights(path)` 读写：

| 键名 | 形状 | 说明 |
|------|------|------|
| `embedding` | (35, 8) | 品种嵌入矩阵（35品种×8维） |
| `fc0.W` | (48, 48) | 第1层全连接权重 |
| `fc0.b` | (48,) | 第1层全连接偏置 |
| `fc1.W` | (48, 32) | 第2层全连接权重 |
| `fc1.b` | (32,) | 第2层全连接偏置 |
| `fc2.W` | (32, 16) | 第3层全连接权重 |
| `fc2.b` | (16,) | 第3层全连接偏置 |
| `fc3.W` | (16, 8) | 第4层全连接权重 |
| `fc3.b` | (8,) | 第4层全连接偏置 |
| `output.W` | (8, 1) | 输出层权重 |
| `output.b` | (1,) | 输出层偏置 |
| `bn0.gamma` / `bn0.beta` | (48,) | 第1层BN缩放/偏移 |
| `bn1.gamma` / `bn1.beta` | (32,) | 第2层BN缩放/偏移 |
| `bn2.gamma` / `bn2.beta` | (16,) | 第3层BN缩放/偏移 |
| `bn3.gamma` / `bn3.beta` | (8,) | 第4层BN缩放/偏移 |
| `bn*.running_mean` / `bn*.running_var` | (*,) | 各层BN运行统计量（推理用） |
| `res_proj.0.W` / `res_proj.0.b` | (48, 32) / (32,) | 残差投影1（48→32） |
| `res_proj.1.W` / `res_proj.1.b` | (32, 16) / (16,) | 残差投影2（32→16） |
| `res_proj.2.W` / `res_proj.2.b` | (16, 8) / (8,) | 残差投影3（16→8） |
| `_hidden_dims` | (4,) | 隐藏层维度 [48, 32, 16, 8] |
| `_input_scores` | (1,) | 输入评分维度 38 |
| `_num_varieties` | (1,) | 品种数量 35 |
| `_embedding_dim` | (1,) | 嵌入维度 8 |
| `_version` | (2,) | 版本号 [4, 0] |

---

## 交易日规则

美股TradFi永续合约特点：

| 维度 | 规则 |
|------|------|
| 美股现货交易 | 周一至周五 9:30-16:00 ET（夏令时13:30-20:00 UTC） |
| **永续合约交易** | **24小时/7天** 无休市（Binance永续不间断） |
| **日切时刻** | **UTC 00:00**（统一日切，对应美股收盘后约4小时） |
| **管线触发** | **UTC 00:15**（日切后15分钟，等待Binance日线K线收线） |
| 周末/节假日 | 永续合约不休市，但现货不交易 → 流动性可能降低 |
| y值计算基准 | 以 **UTC 00:00 价格** 为日切收盘价 |

**美股休市日**（马丁·路德·金日、总统日、耶稣受难日、阵亡将士纪念日、六月节、独立日、劳动节、感恩节、圣诞节）：现货休市但永续合约照常交易，流动性下降可能导致滑点，CatTrader在这些日期应降低仓位或暂停。

---

## 注意事项

1. **冷启动**：训练数据不足25个有效样本时不训练，CANN评分输出0.5（中性）
2. **无权重不推理**：无模型权重文件时统一返回0.5，不使用随机初始化模型
3. **SA评分质量**：CANN精度直接依赖SA评分质量，确保SA按14维度规范执行
4. **过拟合防护**：数据量较小时（<100样本），Dropout和早停法是关键防线
5. **纯NumPy**：不依赖PyTorch或任何深度学习框架，完整手写实现
6. **品种数量**：35个Binance USDT-M TradFi永续品种
7. **时序划分**：训练/验证集按时序80/20划分，不随机打乱（防止未来信息泄露）
8. **数据来源铁律**：D1-D14只能来自SA真实分析，y值只能来自Binance真实涨跌
9. **免责声明**：所有报告必须附带免责声明，本分析仅供参考，不构成投资建议
10. **Binance可用性**：如遇API 451/403，需切换代理；纯requests实现无python-binance依赖
