# 合约分析 (Contract Analysis)

美股 StockAnalysis (US-ST) — 面向美股的14维度分析技能，由A股版本 StockAnalysis(SA) 转换而来。

## 项目目标

将面向A股的14维度分析框架完整迁移到美股市场，适配美国市场的数据源、宏观指标、行业分类和资金流向体系。

## 维度框架 (14 Dimensions)

### 宏观维度 (3维)
- 货币政策 — FOMC / 联邦基金利率 / 缩表(QT) / 点阵图
- 经济周期 — GDP / 非农就业 / CPI-PPI / ISM PMI / 收益率曲线
- 财政政策 — 联邦预算 / 债务上限 / 财政刺激

### 行业维度 (2维)
- 行业景气度 — GICS/ICB 行业分类
- 比较优势 — 行业轮动 / 相对强度

### 公司维度 (3维)
- 盈利能力 — ROE/ROIC/利润率 / FCF Yield
- 估值安全边际 — PE/PB/PS 相对 S&P 500 历史分位
- 盈利预期 — Earnings Surprise / Guidance / Analyst Revision

### 市场维度 (4维)
- 资金流向 — 机构13F持仓 / VIX / Put-Call Ratio
- 机构持仓 — Insider Trading / Institutional Ownership
- 市场情绪 — VIX / CNN Fear & Greed / AAII Sentiment
- 技术结构 — 相对强弱(RS) / MACD / 跨市场信号

### 治理催化维度 (2维)
- 公司治理 — Board Structure / Shareholder Returns (Buyback+Dividend)
- 事件驱动 — SEC Filing / Earnings Call / FDA/Patent / M&A

## 数据源

| A股数据源 | 美股替代数据源 |
|-----------|---------------|
| 新浪财经免费接口 | Yahoo Finance API |
| 东方财富北向资金API | FRED (Federal Reserve Economic Data) |
| — | Finnhub |
| — | Alpha Vantage |
| — | SEC EDGAR |

## 动态权重预设 (6种)

- 牛市初期 (Early Bull)
- 中后期 (Mid-Late Bull)
- 熊市 (Bear Market)
- 财报季 (Earnings Season)
- 政策期 (FOMC Period)
- 机构主导 (Institutional Dominant)
