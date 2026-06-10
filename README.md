# 合约分析 — 美股TradFi永续合约分析交易系统

基于三层架构的 **Binance USDT-M TradFi 永续合约**（美股个股/ETF/商品）量化分析交易技能集。

## 架构总览

```
StockAnalysis (SA)          ← 14维度基本面评分（宏观3+行业2+公司3+市场4+治理催化2）
    │
    ├─ 14维评分 [0,1] + 24维技术面
    │
    ▼
CANN                        ← 神经聚合引擎（纯NumPy）
    │                         14基本面 + 24技术面 → 48维输入
    │                         4层残差MLP → Sigmoid输出 [0,1]
    │
    ├─ 综合评分 s ∈ [0,1]
    │
    ▼
CatTrader                   ← 趋势跟踪交易系统
                              每4h查表 → 0.3×/0.5×杠杆
                              做多/做空/平仓/持有
```

## 技能清单

| 技能 | 目录 | 描述 |
|------|------|------|
| **StockAnalysis (SA)** | `skills/StockAnalysis/` | 14维度美股基本面分析 |
| **CANN** | `skills/CANN/` | 纯NumPy神经网络：48维→评分 |
| **CatTrader** | `skills/CatTrader/` | 趋势跟踪决策：5区间映射，6次/天 |

## 14维度框架

| # | 维度 | 类别 | 核心关注 |
|---|------|------|----------|
| 1 | 货币政策 | 宏观 | FOMC/联邦基金利率/DXY/实际利率 |
| 2 | 经济周期 | 宏观 | GDP/NFP/CPI/ISM PMI/收益率曲线 |
| 3 | 财政政策 | 宏观 | 联邦预算/债务上限/税收政策 |
| 4 | 行业景气度 | 行业 | GICS行业/行业轮动/子行业趋势 |
| 5 | 比较优势 | 行业 | 市场份额/护城河/竞争格局 |
| 6 | 盈利能力 | 公司 | ROE/ROIC/利润率/FCF/营收增长 |
| 7 | 估值安全边际 | 公司 | PE/PB/PS分位/PEG/EV-EBITDA |
| 8 | 盈利预期 | 公司 | Earnings Surprise/Guidance/Analyst |
| 9 | 合约资金流 | 市场 | OI+价格组合/资金费率(Binance API) |
| 10 | 机构动向 | 市场 | ETF流量/13F/内部人/做空比例 |
| 11 | 市场情绪 | 市场 | VIX/Put-Call/AAII/Fear & Greed |
| 12 | 技术结构 | 市场 | MA排列/布林带/ATR/RS vs SPY (Binance K线) |
| 13 | 公司治理 | 治理 | Board/回购+分红/高管薪酬 |
| 14 | 事件驱动 | 催化 | Earnings Call/FDA/产品发布/并购 |

## 品种覆盖（35个Binance TradFi永续）

| 类别 | 品种 |
|------|------|
| 科技巨头 (7) | NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA |
| 半导体 (4) | INTC, AMD, AVGO, QCOM |
| 加密相关 (5) | MSTR, COIN, CRCL, HOOD, PLTR |
| 企业科技 (4) | ORCL, CSCO, UBER, SOFI |
| 消费零售 (3) | DIS, HD, SBUX |
| 医药 (2) | LLY, NVS |
| ETF (5) | SPY, QQQ, SOXL, GLD, IBIT |
| 商品 (3) | XAU, XAG, CL |
| Pre-IPO (2) | SPACEX, OPENAI |

## 数据源

| 数据 | 来源 |
|------|------|
| K线/价格/OI/资金费率 | Binance API (免费) |
| 公司财务/估值 | Yahoo Finance + SEC EDGAR |
| 宏观指标 | FRED + Web Search |
| ETF流量/13F/内部人 | SEC EDGAR + Fintel |
| VIX/Put-Call/情绪 | CBOE + AAII |

## 交易调度

CatTrader 每4小时决策一次（6次/天 UTC）：

| UTC | 00:00 | 04:00 | 08:00 | 12:00 | 16:00 | 20:00 |
|-----|-------|-------|-------|-------|-------|-------|

## 快速启动

```bash
# 1. 安装依赖
pip install python-binance numpy pandas

# 2. 测试Binance数据
cd skills/StockAnalysis/scripts
python binance_data.py

# 3. CANN管线
cd skills/CANN/scripts
python daily_pipeline.py --date 20260610

# 4. CatTrader决策
cd skills/CatTrader/scripts
python cattrader.py
```

## 目录结构

```
skills/
├── StockAnalysis/          # 14维度美股基本面分析
│   ├── SKILL.md
│   ├── references/         # 维度手册 + 报告模板
│   └── scripts/            # binance_data.py
├── CANN/                   # 神经网络评分
│   ├── SKILL.md
│   ├── references/         # 品种列表
│   ├── scripts/            # pretrain_numpy.py + daily_pipeline.py
│   └── data/               # 训练数据 + 模型权重
├── CatTrader/              # 交易决策
│   ├── SKILL.md
│   ├── scripts/cattrader.py
│   └── data/               # 状态 + 报告
└── common/                 # 公共模块
    └── variety_list.py     # 统一品种映射
```

## 参考

- 原始商品期货版技能（A股CA/CANN/CatTrader）位于 `_app_data_` 目录
- 本文档面向Binance USDT-M TradFi永续合约（美股/ETF/商品）

## 免责声明

本系统仅供分析和研究参考，不构成投资建议。永续合约交易具有高风险，可能导致全部本金甚至超额损失。请根据自身风险承受能力审慎决策。
