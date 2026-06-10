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
- StockAnalysis (SA) 技能：提供14维度基本面评分
- 技术面：`skills/StockAnalysis/scripts/binance_data.py` 提供24维度技术面评分
- NumPy：纯NumPy实现

## ⚠️ 铁律：禁止合成数据
- D1-D14 **必须**来自SA技能真实分析
- T1-T24 **必须**来自Binance API真实K线计算
- y值**必须**来自Binance API真实次日涨跌（sigmoid映射）
- **禁止**用任何公式推算/合成y值

---

## 神经网络架构 v4.0

### 输入（48维）
| 序号 | 输入 | 类型 | 编码后维度 |
|------|------|------|-----------|
| 1-14 | SA基本面评分 | float [0,1] | 14 |
| 15-38 | 技术面评分 | float [0,1] | 24 |
| 39-40 | 月份编码 | sin/cos | 2 |
| 41-48 | 品种嵌入 | Embedding(dim=8) | 8 |

### 网络结构
```
Input(48) → Linear(48,48)+BN+ReLU+Drop(0.25)+Residual
          → Linear(48,32)+BN+ReLU+Drop(0.25)+ResidualProj
          → Linear(32,16)+BN+ReLU+Drop(0.25)+ResidualProj
          → Linear(16,8)+BN+ReLU+Drop(0.25)+ResidualProj
          → Linear(8,1) → Sigmoid → [0,1]
```

### 超参数
| 参数 | 值 | 说明 |
|------|-----|------|
| Loss | MSE | 回归 [0,1] |
| Optimizer | Adam (手写) | lr=5e-5, embedding_lr×5 |
| Batch | 32 | |
| Epochs | 100 | EarlyStopping patience=15 |
| Train/Val | 80/20 | 时序划分 |

---

## 每日执行流程（UTC 00:15）

1. **回填昨日y值**：Binance API次日涨跌→sigmoid映射更新historical_samples.csv
2. **微调模型**：仅用有效样本（真实SA+真实y）
3. **技术面采集**：自动从Binance K线计算24维
4. **追加今日样本**：SA评分写入CSV，y=0.5占位
5. **全品种推理**：50→35品种（TradFi股票+ETF）
6. **生成报告**

---

## 品种数量
**35个** Binance TradFi USDT-M 永续品种（美股个股32+ETF5+商品3+Pre-IPO2，可能随Binance上架变化）。

## 冷启动
- <25有效样本 → 不训练，CANN=0.5（中性）
- 25-100样本 → 高Dropout(0.35)防过拟合
- >100样本 → 标准训练

## 交易日规则
- 美股现货：周一至周五 9:30-16:00 ET
- 永续合约：24/7交易
- 日切：UTC 00:00
- 管线时间：UTC 00:15

## 注意事项
- CANN精度依赖SA评分质量
- 无模型权重→统一返回0.5（中性）
- 免责声明：本分析仅供参考，不构成投资建议
