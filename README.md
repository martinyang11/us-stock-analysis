# 合约分析 — gTrade TradFi 永续合约分析交易系统

基于三层架构的 **Gains Network gTrade** 去中心化永续合约（美股个股/ETF）量化分析交易系统。

## 架构总览

```
StockAnalysis (SA)          ← 14维度基本面评分
    │                          D1-D3 宏观 + D4-D5 行业 + D6-D8 公司
    │                          D9-D12 市场 + D13-D14 治理催化
    │
    ├─ 14维评分 + 24维技术面
    │
    ▼
SANN                        ← 神经聚合引擎（纯NumPy v4.0）
    │                         48维输入 → 4层残差MLP → Sigmoid [0,1]
    │
    ├─ 综合评分 s ∈ [0,1]
    │
    ▼
CatTrader                   ← 趋势跟踪交易系统
                              5区间映射 → 0.3×/0.5×杠杆
                              多/空/平/持 + 15%止损止盈
```

## 技能清单

| 技能 | 目录 | 描述 |
|------|------|------|
| **StockAnalysis (SA)** | `skills/StockAnalysis/` | 14维度美股基本面分析 |
| **SANN** | `skills/SANN/` | 纯NumPy神经网络：48维→评分 v4.0 |
| **CatTrader** | `skills/CatTrader/` | 趋势跟踪决策：5区间映射 + 链上执行 |

## 品种覆盖（29个 gTrade TradFi）

| 类别 | 数量 | 品种 |
|------|------|------|
| 科技巨头 | 7 | NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA |
| 加密相关 | 5 | MSTR, COIN, HOOD, MARA, RIOT |
| 科技其他 | 5 | NFLX, SNAP, PLTR, PYPL, CRCL |
| 消费零售 | 1 | MCD |
| 工业/国防 | 1 | LMT |
| 模因 | 1 | GME |
| ETF/指数 | 7 | SPY, QQQ, IWM, DIA, GDX, URA, URNM |
| 矿业 | 1 | WPM |
| 航天 | 1 | SPCX |

## 14维度框架

| # | 维度 | 类别 | 数据窗口 |
|---|------|------|----------|
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

## SANN v4.0 架构

- **输入**: 14 SA基本面 + 24 技术面 + 2 月份编码 + 8 品种嵌入 = **48维**
- **网络**: 4层残差MLP (48→32→16→8→1) + BatchNorm + Dropout(0.25)
- **输出**: Sigmoid → s ∈ [0,1]
- **训练**: Adam优化器(手写NumPy), MSE损失, 早停patience=15
- **冷启动**: <25有效样本不训练, 输出0.5中性

## CatTrader 仓位映射

```
s:  0 ════ 0.35 ════ 0.45 ════ 0.55 ════ 0.65 ════ 1
    │空0.5× │ 空0.3×  │  平仓   │ 多0.3×  │多0.5× │
```

## 自动化调度

| 时间 (北京) | 任务 | 频率 |
|-------------|------|------|
| **20:57** | SA 评分（Web Research + 脚本） | 周一至周五 |
| **21:07** | CatTrader 交易决策 | 周一至周五 |

通过 Claude Code cron 自动执行，持久化于 `.claude/scheduled_tasks.json`。

## 数据源

| 数据 | 来源 |
|------|------|
| 品种元数据/Spread | gTrade REST API (`backend-arbitrum.gains.trade`) |
| 实时价格 | gTrade WebSocket v4 (`backend-pricing.eu.gains.trade`) |
| 历史K线/财务 | Yahoo Finance (yfinance) |
| 宏观/情绪 | Web Search + FRED + CBOE |
| 链上交易 | Gains Diamond v10 (Arbitrum) |

## 保护机制

| 机制 | 规则 |
|------|------|
| 止损 | 15%（做多：entry × 0.85，做空：entry × 1.15） |
| 止盈 | 15%（做多：entry × 1.15，做空：entry × 0.85） |
| Spread保护 | > 0.6% → 仓位降一档 |
| 中性区保护 | s ∈ (0.45, 0.55) 绝不开仓 |
| 链上预检 | eth_call 预检 → 失败自动跳过 |
| 状态同步 | 每次链上操作后同步 state.json |

## 快速启动

```bash
# 1. 代理 (上海/香港需要)
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897

# 2. 安装依赖
pip install numpy pandas requests yfinance websocket-client

# 3. SA 评分
.venv/bin/python3 scripts/run_sa_scoring.py --date 20260625

# 4. CatTrader 决策
.venv/bin/python3 skills/CatTrader/scripts/cattrader.py --date 20260625
```

## 目录结构

```
.
├── scripts/
│   ├── run_sa_scoring.py              # SA评分主脚本
│   └── backfill_sa_sann_history.py    # 历史数据回填
├── skills/
│   ├── StockAnalysis/                 # 14维度美股基本面分析
│   │   ├── SKILL.md
│   │   └── scripts/gtrade_data.py     # gTrade REST + WebSocket + yfinance
│   ├── SANN/                          # 神经网络评分 v4.0
│   │   ├── SKILL.md
│   │   ├── scripts/
│   │   │   ├── pretrain_numpy.py      # 模型定义+训练+推理
│   │   │   └── daily_pipeline.py      # 每日管线
│   │   └── data/
│   │       ├── daily_scores/          # scores_YYYYMMDD.csv
│   │       ├── model_weights.npz      # 模型权重
│   │       └── historical_samples.csv # 训练数据
│   ├── CatTrader/                     # 交易决策
│   │   ├── SKILL.md
│   │   ├── scripts/cattrader.py
│   │   └── data/state.json            # 持仓状态
│   └── common/
└── onchain_trade/                     # 链上交易适配器
    └── hole_board/exchange/onchain/venues/gains/
        └── adapter.py                 # Gains Diamond v10
```

## 链上交易

- **合约**: Gains v10 Diamond (Arbitrum) `0xFF162c694eAA571f685030649814282eA457f169`
- **保证金**: USDC (10 USDC最低)
- **杠杆**: 0.3× / 0.5×
- **预检**: eth_call 模拟 → 失败自动跳过

## 免责声明

本系统仅供分析和研究参考，不构成投资建议。永续合约交易具有高风险，可能导致全部本金甚至超额损失。请根据自身风险承受能力审慎决策。
