# 合约分析 — gTrade TradFi 永续合约分析交易系统

基于三层架构的 **Gains Network gTrade** 去中心化永续合约（美股个股/ETF/商品/加密）量化分析交易技能集。

## 架构总览

```
StockAnalysis (SA)          ← 14维度基本面评分（宏观3+行业2+公司3+市场4+治理催化2）
    │
    ├─ 14维评分 [0,1] + 24维技术面
    │
    ▼
SANN                        ← 神经聚合引擎（纯NumPy）
    │                         14基本面 + 24技术面 → 48维输入
    │                         4层残差MLP → Sigmoid输出 [0,1]
    │
    ├─ 综合评分 s ∈ [0,1]
    │
    ▼
CatTrader                   ← 趋势跟踪交易系统
                              美股时段4次/天查表 → 0.3×/0.5×杠杆
                              做多/做空/平仓/持有 + 15%止损止盈
```

## 技能清单

| 技能 | 目录 | 描述 |
|------|------|------|
| **StockAnalysis (SA)** | `skills/StockAnalysis/` | 14维度美股基本面分析 |
| **SANN** | `skills/SANN/` | 纯NumPy神经网络：48维→评分 |
| **CatTrader** | `skills/CatTrader/` | 趋势跟踪决策：5区间映射 + SL/TP |

## 14维度框架

| # | 维度 | 类别 | 核心关注 | 数据源 |
|---|------|------|----------|--------|
| 1 | 货币政策 | 宏观 | FOMC/联邦基金利率/DXY/实际利率 | Web Search + FRED |
| 2 | 经济周期 | 宏观 | GDP/NFP/CPI/ISM PMI/收益率曲线 | Web Search + FRED |
| 3 | 财政政策 | 宏观 | 联邦预算/债务上限/税收政策 | Web Search |
| 4 | 行业景气度 | 行业 | GICS行业/行业轮动/子行业趋势 | Web Search + Yahoo Finance |
| 5 | 比较优势 | 行业 | 市场份额/护城河/竞争格局 | Web Search + SEC EDGAR |
| 6 | 盈利能力 | 公司 | ROE/ROIC/利润率/FCF/营收增长 | Yahoo Finance + SEC EDGAR |
| 7 | 估值安全边际 | 公司 | PE/PB/PS分位/PEG/EV-EBITDA | Yahoo Finance |
| 8 | 盈利预期 | 公司 | Earnings Surprise/Guidance/Analyst | Yahoo Finance + Seeking Alpha |
| 9 | 合约资金流 | 市场 | gTrade Spread + 24h成交量 + 市场开关 | gTrade API + yfinance |
| 10 | 机构动向 | 市场 | ETF流量/13F/内部人/做空比例 | SEC EDGAR + Web Search |
| 11 | 市场情绪 | 市场 | VIX/Put-Call/AAII/Fear & Greed | CBOE + AAII |
| 12 | 技术结构 | 市场 | MA排列/布林带/ATR/RSI (yfinance K线) | yfinance → numpy计算 |
| 13 | 公司治理 | 治理 | Board/回购+分红/高管薪酬 | SEC EDGAR + Web Search |
| 14 | 事件驱动 | 催化 | Earnings Call/FDA/产品发布/并购 | Web Search |

## 品种覆盖（56个 gTrade TradFi）

| 类别 | 数量 | 品种 |
|------|------|------|
| 科技巨头 | 7 | NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA |
| 半导体 | 2 | AMD, INTC |
| 加密相关 | 6 | MSTR, COIN, CRCL, HOOD, MARA, RIOT |
| 金融科技 | 3 | V, MA, PYPL |
| 消费零售 | 8 | DIS, NKE, KO, MCD, WMT, SBUX, ABNB, GME |
| 科技其他 | 7 | NFLX, SNAP, PLTR, SBET, BIDU, ROKU, WPM |
| 医药 | 1 | PFE |
| 工业/国防 | 2 | BA, LMT |
| ETF/指数 | 10 | SPY, QQQ, IWM, DIA, SPX500, NAS100, USA30, GDX, URA, URNM |
| 商品 | 8 | XAU, XAG, WTI, XPT, XPD, HG, NATGAS, BRENT |
| 加密 | 2 | BTC, ETH |

## 数据源

| 数据 | 来源 |
|------|------|
| 品种元数据/Spread/市场状态 | gTrade REST API (`backend-arbitrum.gains.trade`) |
| 实时 Mark 价格 | gTrade WebSocket v4 (`backend-pricing.eu.gains.trade`) |
| 历史 K线 | yfinance (gTrade 合成价格跟踪标的现货) |
| 公司财务/估值 | Yahoo Finance + SEC EDGAR |
| 宏观指标 | FRED + Web Search |
| VIX/Put-Call/情绪 | CBOE + AAII |

## 交易调度

CatTrader 美股时段 4 次/天（周一至周五），通过 crontab 自动执行：

| 轮次 | UTC | 北京时间 | 美股 |
|------|-----|---------|------|
| 早盘 | 14:00 | 22:00 | 开盘+30min |
| 午前 | 16:30 | 00:30 | 盘中 |
| 午后 | 18:30 | 02:30 | 尾盘前 |
| 收盘 | 20:30 | 04:30 | 收盘决策 |

SANN 每日管线 UTC 21:00（北京时间 05:00）：回填 y 值 → 微调模型 → 全品种推理。

## 保护机制

| 机制 | 规则 |
|------|------|
| 止损 | 15%（做多：entry × 0.85，做空：entry × 1.15） |
| 止盈 | 15%（做多：entry × 1.15，做空：entry × 0.85） |
| 市场关闭 | `isStocksOpen=false` → 不新开仓 |
| 高 Spread | > 0.6% → 仓位降一档 |
| 中性区 | s ∈ (0.45, 0.55) 绝不开仓 |

## 快速启动

```bash
# 1. 安装依赖
pip install numpy pandas requests yfinance websocket-client

# 2. 代理 (上海需要)
export HTTPS_PROXY=http://127.0.0.1:7897

# 3. 测试 gTrade 数据
cd skills/StockAnalysis/scripts
python gtrade_data.py

# 4. SANN 管线
cd skills/SANN/scripts
python daily_pipeline.py --date 20260612 --data-dir ../data

# 5. CatTrader 决策
cd skills/CatTrader/scripts
python cattrader.py

# 6. 安装自动化调度
crontab crontab.txt
```

## 目录结构

```
skills/
├── StockAnalysis/              # 14维度美股基本面分析
│   ├── SKILL.md
│   ├── references/             # 维度手册 + 报告模板
│   └── scripts/
│       └── gtrade_data.py      # gTrade REST + WebSocket + yfinance
├── SANN/                       # 神经网络评分 (原 CANN)
│   ├── SKILL.md
│   ├── references/
│   ├── scripts/
│   │   ├── pretrain_numpy.py
│   │   └── daily_pipeline.py
│   └── data/
├── CatTrader/                  # 交易决策
│   ├── SKILL.md
│   ├── scripts/cattrader.py
│   └── data/
└── common/
    └── variety_list.py         # gTrade 56品种动态映射
```

## 免责声明

本系统仅供分析和研究参考，不构成投资建议。永续合约交易具有高风险，可能导致全部本金甚至超额损失。请根据自身风险承受能力审慎决策。
