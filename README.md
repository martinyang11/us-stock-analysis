# 合约分析 — 加密永续合约分析交易系统

基于三层架构的 **Binance USDT-M 永续合约** 量化分析交易技能集。

## 架构总览

```
CryptoAnalysis (CA)         ← 14维度基本面评分（D1-D14）
    │
    ├─ 14维评分 [0,1]
    │
    ▼
CANN                        ← 神经聚合引擎
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
| **CryptoAnalysis** | `skills/CryptoAnalysis/` | 14维度加密基本面分析：全球流动性/地缘政治/监管政策/宏观数据/加密情绪/宏观流动性/链上供应/网络需求/资金费率/跨市场联动/价格位置/加密周期/合约资金流/期限结构 |
| **CANN** | `skills/CANN/` | 纯NumPy神经网络：48维输入→4层残差MLP→综合评分。监督学习，每日微调 |
| **CatTrader** | `skills/CatTrader/` | 趋势跟踪决策：5区间仓位映射，6次/天，资金费率保护，OI过热警告 |

## 数据源

| 数据 | 来源 | 说明 |
|------|------|------|
| K线/OHLCV | Binance API | 免费，REST接口 |
| 资金费率 | Binance API | USDT-M永续 |
| 持仓量/OI | Binance API | 实时+历史 |
| 多空比 | Binance API | 需要API Key |
| 交割合约 | Binance API | 期限结构计算 |
| Fear & Greed | Alternative.me | 免费 |
| 币种元数据 | CoinGecko API | 免费 |
| 宏观数据 | FRED / Web Search | 免费 |
| 链上数据 | CoinGecko / CryptoQuant | 免费/付费 |

## 14维度框架

| # | 维度 | 类别 | 核心指标 |
|---|------|------|----------|
| 1 | 全球流动性 | 宏观 | FOMC / DXY / 实际利率 |
| 2 | 地缘政治 | 宏观 | 冲突 / 制裁 / BTC避险叙事 |
| 3 | 监管政策 | 宏观 | SEC / ETF流量 / 司法辖区 |
| 4 | 宏观数据 | 宏观 | CPI / NFP / GDP / ISM PMI |
| 5 | 加密情绪 | 情绪 | Fear & Greed / 资金费率聚合 |
| 6 | 宏观流动性 | 宏观 | 全球M2 / QT / 稳定币市值 |
| 7 | 链上供应 | 链上 | 交易所余额 / 鲸鱼 / 代币解锁 |
| 8 | 网络需求 | 链上 | 活跃地址 / TVL / 用户增长 |
| 9 | 资金费率/基差 | 合约 | 资金费率 / 费率分位数 |
| 10 | 跨市场联动 | 市场 | BTC-Nasdaq / ETH/BTC / BTC.D |
| 11 | 价格位置 | 技术 | MA20/50/200 / 布林带 / ATR |
| 12 | 加密周期 | 周期 | 减半周期 / 山寨季 / 4年周期 |
| 13 | 合约资金流 | 合约 | OI+价格组合 / 多空比 / 清算地图 |
| 14 | 期限结构 | 合约 | Contango/Backwardation / 跨期价差 |

## 币种覆盖

**Top 50 市值币种**，分6大类别：
- **L1** (16): BTC, ETH, BNB, SOL, XRP, ADA, AVAX, DOT, LTC, ATOM, NEAR, APT, SUI, ICP, SEI, ETC
- **L2** (7): MATIC, OP, ARB, STX, STRK, ZK, MOVE
- **DeFi** (11): UNI, LINK, MKR, AAVE, INJ, ENA, JUP, CRV, LDO, ONDO, PENDLE
- **Meme** (7): DOGE, SHIB, PEPE, WIF, BONK, FLOKI, BRETT
- **AI** (4): RENDER, TAO, FET, WLD
- **Infra** (5): TIA, PYTH, FIL, GRT, W

## 交易调度

加密市场24/7，CatTrader每4小时决策一次（6次/天 UTC）：

| UTC | 00:00 | 04:00 | 08:00 | 12:00 | 16:00 | 20:00 |
|-----|-------|-------|-------|-------|-------|-------|
| CST | 08:00 | 12:00 | 16:00 | 20:00 | 00:00 | 04:00 |

## 快速启动

```bash
# 1. 安装依赖
pip install python-binance numpy pandas

# 2. 测试Binance数据接口
cd skills/CryptoAnalysis/scripts
python binance_data.py

# 3. 运行CANN每日管线（需先有CA评分）
cd skills/CANN/scripts
python daily_pipeline.py --date 20260610

# 4. 运行CatTrader决策
cd skills/CatTrader/scripts
python cattrader.py
```

## 免责声明

本系统仅供分析和研究参考，不构成投资建议。加密货币交易具有极高风险，可能导致全部本金甚至超额损失。请根据自身风险承受能力审慎决策。
