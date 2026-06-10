# CANN (CommodityAnalysis Neural Network) - 商品期货神经网络评分技能

## 描述
基于 CA 技能14维度基本面评分 + 24维度技术面评分，通过纯NumPy神经网络输出综合多空评分。网络通过真实CA评分与实际涨跌的对应关系进行监督学习，逐步优化评分精度。

## 简称
CANN

## 触发场景
- 用户说"CANN"、"CANN分析"、"神经网络评分"等
- 日程触发（标题含"CANN"或"CA-NN"）
- 每个交易日 16:00 自动执行

## 依赖
- CommodityAnalysis (CA) 技能：提供14维度基本面评分（**必须来自真实分析**）
- 技术面模块：`skills/common/technical_indicators.py` 提供24维度技术面评分
- NumPy：纯NumPy实现，无PyTorch/深度学习框架依赖

## ⚠️ 铁律：禁止合成数据

- D1-D14（基本面评分）**必须**来自CA技能的真实14维度分析输出
- T1-T24（技术面评分）**必须**来自TQSDK/AKShare真实K线计算
- y值**必须**来自AKShare真实次日涨跌（sigmoid映射）
- **禁止**用任何公式推算/合成y值或其他维度
- 有多少天真实CA数据，就有多少天有效训练样本

---

## 神经网络架构 v4.0

### 输入（48维）

| 序号 | 输入 | 类型 | 编码方式 | 编码后维度 |
|------|------|------|----------|-----------|
| 1-14 | CA基本面评分 | float [0,1] | 直接输入 | 14 |
| 15-38 | 技术面评分 | float [0,1] | 直接输入 | 24 |
| 39-40 | 月份 | int [1,12] | 周期编码 sin/cos | 2 |
| 41-48 | 品种ID | int [0,52] | Embedding(dim=8) | 8 |

**月份周期编码**：`month_sin = sin(2π × month / 12)`，`month_cos = cos(2π × month / 12)`，捕捉月份的周期性（12月与1月相邻）。

**品种ID嵌入**：通过 Embedding 层学习每个品种的潜在特征表示，维度为8。

**有效输入总维度**：14 + 24 + 2 + 8 = **48**

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
| 4层隐藏层（48→32→16→8）+ 残差连接 | 适度深度，残差连接防止梯度消失，提升收敛 |
| BatchNorm（手写NumPy） | 加速收敛，缓解小批量训练的不稳定 |
| Dropout(0.25) | 防止过拟合，适度正则 |
| Sigmoid输出 | 输出范围严格 [0,1]，与评分语义一致 |
| Embedding(8) | 53个品种，8维嵌入足以区分品种特征 |
| 月份sin/cos | 保留周期性，避免线性编码破坏12月≈1月的关系 |
| 纯NumPy实现 | 无外部依赖，沙箱环境兼容 |

### 损失函数

| 损失项 | 权重 | 说明 |
|--------|------|------|
| MSE | 1.0 | 主损失，回归任务 |
| 软标签BCE | 0.0（已禁用） | 梯度是MSE的100+倍，导致不收敛 |
| 对比学习 | 0.0（已禁用） | 梯度注入与主损失Adam更新冲突 |

### 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| Loss | MSE | 回归任务，输出连续值[0,1] |
| Optimizer | Adam（手写NumPy） | 自适应学习率，支持分层LR |
| Learning Rate | 5e-5 | 微调学习率（embedding层lr×5.0） |
| Batch Size | 32 | 微调批量 |
| Epochs | 100 | 最大训练轮数 |
| Early Stopping Patience | 15 | 验证集连续15轮不改善则停止 |
| Train/Val Split | 80/20 | 时序划分（前80%训练，后20%验证） |
| Gradient Clip Norm | 1.0 | 全局梯度裁剪，防止梯度爆炸 |

---

## 每日执行流程

### 第一步：采集技术面（自动）

通过TQSDK/AKShare获取53品种K线数据，计算24维技术面评分：
```python
from skills.common.technical_indicators import compute_from_klines
tech_scores = compute_from_klines(klines)  # 返回24维[0,1]评分
```

TQSDK优先，AKShare降级。技术面数据写入 `daily_scores/scores_YYYYMMDD.csv`，D1-D14列暂填-1（待CA填充）。

### 第二步：CA评分采集（必须手动执行）

逐品种调用CA技能获取真实14维评分，然后写入CSV：
```python
from skills.CANN.scripts.daily_pipeline import write_ca_scores
write_ca_scores('YYYYMMDD', variety_id, [d1,...,d14], './CANN/data/daily_scores')
# 输入校验：品种ID 0-52，CA评分14维且每维在[0,1]
```

⚠️ CA评分必须来自真实分析，禁止合成/生成。53品种约需30-60分钟。

### 第三步：计算真实涨跌标签 y 值

在下一个交易日结束时，用AKShare获取该品种次日真实涨跌幅：
```python
ret = (next_close - today_close) / today_close
y = 1 / (1 + exp(-ret * 20))  # sigmoid映射到[0,1]
```

| 涨跌幅 ret | y值 | 含义 |
|------------|------|------|
| +3% | 0.95 | 强利多 |
| +1% | 0.73 | 偏多 |
| 0% | 0.50 | 中性 |
| -1% | 0.27 | 偏空 |
| -3% | 0.05 | 强利空 |

**y值占位规则**：当日样本的y值用0.5占位（raw_change=0标记），次日自动回填真实涨跌。**禁止用公式推算y值。**

### 第四步：训练神经网络

仅使用**同时具备真实CA+真实y值**的有效样本训练（`load_csv_samples(filter_invalid=True)`自动过滤dim<0和raw_change=0的行）。

**训练策略**：
- 从 `CANN/data/historical_samples.csv` 加载有效历史数据
- 将当日新样本追加到历史数据
- 全量有效数据微调网络（小学习率+分层LR+时序划分）
- 使用早停法防止过拟合
- 保存模型权重到 `CANN/data/model_weights.npz`
- 保存训练日志

**冷启动**：有效样本<30条时不训练，推理结果为0.5（中性）。无权重文件时不使用随机模型。

### 第五步：推理评分

用训练好的网络对53品种计算综合评分：
```python
from skills.CANN.scripts.pretrain_numpy import predict_single
cann_score = predict_single(model, dim_scores_14d, month, variety_id, tech_scores=tech_scores_24d)
```

- 输入：14维CA评分 + 24维技术面评分 + 当前月份 + 品种ID
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

在一张总表中列出所有品种的：
- 14维CA评分
- 24维技术面评分
- CA加权综合评分
- CANN综合评分
- 评分差异（CANN - CA）
- 多空研判

报告保存到 `CANN/reports/CANN报告_YYYYMMDD.md`

---

## 数据存储结构

```
./CANN/
├── data/
│   ├── historical_samples.csv      # 累积训练数据（仅含有效样本：真实CA+真实y）
│   ├── daily_scores/               # 每日评分（技术面自动+CA手动）
│   │   ├── scores_20260602.csv     # 格式：date,variety_id,variety_name,month,dim1..14,tech1..24
│   │   └── ...
│   ├── model_weights.npz           # 当前模型权重
│   └── training_log.csv            # 训练日志
├── reports/
│   ├── CANN报告_20260602.md
│   └── ...
└── scripts/
    ├── pretrain_numpy.py           # 核心脚本：模型定义+训练+推理
    └── daily_pipeline.py           # 每日管线：技术面采集→CA写入→y值回填→微调→推理
```

### CSV格式示例（daily_scores/scores_YYYYMMDD.csv）

```
date,variety_id,variety_name,month,dim1,dim2,...,dim14,tech1,tech2,...,tech24
2026-06-02,7,沪铜,6,0.6,0.5,0.65,0.55,0.45,0.6,0.4,0.55,0.5,0.55,0.5,0.45,0.6,0.72,0.35,...
```

- dim1-dim14：CA评分[0,1]，未填充时为-1
- tech1-tech24：技术面评分[0,1]

### historical_samples.csv 额外列

```
...,y,raw_change
...,0.731058,0.012000    ← 有效样本（raw_change≠0）
...,0.500000,0.000000    ← 待回填（raw_change=0，训练时自动过滤）
```

---

## 权重文件格式（v4.0）

模型权重保存为 `.npz` 格式，键名规范：

| 键名 | 形状 | 说明 |
|------|------|------|
| `embedding` | (53, 8) | 品种嵌入矩阵 |
| `fc{i}.W` | (in, out) | 第i层全连接权重 |
| `fc{i}.b` | (out,) | 第i层全连接偏置 |
| `bn{i}.gamma` | (out,) | 第i层BN缩放 |
| `bn{i}.beta` | (out,) | 第i层BN偏移 |
| `bn{i}.running_mean` | (out,) | 第i层BN运行均值 |
| `bn{i}.running_var` | (out,) | 第i层BN运行方差 |
| `res_proj.{j}.W` | (in, out) | 残差投影权重 |
| `res_proj.{j}.b` | (out,) | 残差投影偏置 |
| `output.W` | (8, 1) | 输出层权重 |
| `output.b` | (1,) | 输出层偏置 |
| `_hidden_dims` | (4,) | 隐藏层维度 [48,32,16,8] |
| `_input_scores` | (1,) | 输入评分维度数 38 |
| `_version` | (2,) | 版本号 [4,0] |

---

## 交易日判断规则

中国商品期货交易日规则：
1. 周六、周日为非交易日
2. 法定节假日为非交易日（元旦、春节、清明、劳动节、端午、中秋、国庆）
3. 节假日调休补班日：若交易所公告休市，则仍为非交易日
4. 不确定时，通过搜索"上期所/大商所/郑商所 交易日历"确认

---

## 注意事项

1. **冷启动**：训练数据不足30个有效样本时，不进行训练，CANN评分输出为0.5（中性）
2. **无权重不推理**：无模型权重时不使用随机初始化模型推理，统一返回0.5
3. **CA评分质量**：CANN精度直接依赖CA评分质量，确保CA按规范执行
4. **过拟合防护**：数据量较小时（<500样本），Dropout和早停法是关键防线
5. **纯NumPy**：不依赖PyTorch或任何深度学习框架
6. **免责声明**：所有报告必须附带免责声明，本分析仅供参考，不构成投资建议
7. **品种数量**：53个国内商品期货品种
