# StockAnalysis (SA) — 美股TradFi永续合约14维度分析技能

## 描述
从美股分析师的视角，基于**14维度评估框架**，对 Binance USDT-M 永续合约上的传统金融（TradFi）品种进行系统性分析，覆盖宏观、行业、公司、市场、治理催化五大板块，输出多空研判综合打分（0-1之间）。

**品种范围**：Binance 美股个股永续 + ETF永续 + 商品永续（USDT本位）。

## 触发场景
- 用户要求对某个TradFi永续品种进行基本面+技术面分析
- 用户提到"美股分析"、"永续合约分析"、"TradFi打分"等关键词
- 日程触发（标题含"StockAnalysis"或"美股永续分析"）

## 绝对时间基准（每次执行前必须强制执行）

当前系统时间为：`{{CURRENT_DATETIME_UTC}}`。

所有时间相关的窗口期根据数据类型差异化：
- **实时数据**（价格、OI、资金费率）：4小时窗口
- **日度数据**（K线、技术指标）：24小时窗口
- **周度数据**（ETF流量、OI历史）：7天窗口
- **月度数据**（宏观指标、财报）：30天窗口

---

## 14维度评估框架

| # | 维度 | 类别 | 说明 |
|---|------|------|------|
| 1 | 货币政策 | 宏观 | FOMC决议/联邦基金利率/缩表QT/DXY/实际利率 |
| 2 | 经济周期 | 宏观 | GDP/NFP/CPI/ISM PMI/收益率曲线/消费者信心 |
| 3 | 财政政策 | 宏观 | 联邦预算/债务上限/税收政策/财政刺激 |
| 4 | 行业景气度 | 行业 | GICS行业指数/行业轮动/子行业趋势 |
| 5 | 比较优势 | 行业 | 市场份额/护城河/竞争格局/行业排名 |
| 6 | 盈利能力 | 公司 | ROE/ROIC/利润率/FCF Yield/营收增长趋势 |
| 7 | 估值安全边际 | 公司 | PE/PB/PS vs 行业+历史分位/PEG/EV-EBITDA |
| 8 | 盈利预期 | 公司 | Earnings Surprise/Guidance/Analyst Revision趋势 |
| 9 | 合约资金流 | 市场 | OI变化+价格组合/资金费率方向/大额转账 |
| 10 | 机构动向 | 市场 | ETF净流量/13F持仓/内部人交易/做空比例 |
| 11 | 市场情绪 | 市场 | VIX/Put-Call Ratio/AAII Sentiment/Fear & Greed |
| 12 | 技术结构 | 市场 | MA多空排列/布林带/ATR/RS vs SPY/量价关系 |
| 13 | 公司治理 | 治理 | Board结构/股东回报(回购+分红)/高管薪酬 |
| 14 | 事件驱动 | 催化 | Earnings Call/FDA/产品发布/并购/SEC Filing |

---

## 分析评估步骤

### 第一步：信息和数据采集

**采集方法**：
- **Binance API（优先）**：K线/价格/OI/资金费率 → `binance_data.py`
- **Yahoo Finance**：公司财务数据、估值指标
- **Web Search**：宏观数据、新闻、SEC EDGAR、ETF流量
- **FRED**：宏观经济指标

**数据源策略**：
```
维度1-3（宏观）：Web Search + FRED
维度4-5（行业）：Web Search + Yahoo Finance
维度6-8（公司）：Yahoo Finance + SEC EDGAR + Web Search
维度9、12（永续特有）：Binance API ← 优先
维度10-11（市场-传统）：Web Search + Yahoo Finance
维度13-14（治理催化）：Web Search + SEC EDGAR
```

Binance API 使用方式：
```python
from skills.StockAnalysis.scripts.binance_data import BinanceDataProvider

with BinanceDataProvider() as provider:
    klines = provider.get_klines("NVDAUSDT", interval="1h", limit=200)
    funding = provider.get_funding_rate("NVDAUSDT")
    oi = provider.get_open_interest("NVDAUSDT")
    indicators = provider.get_technical_indicators("NVDAUSDT")
```

**全局要求**：

1. **时间精确性**：所有时间必须精确到分钟。若仅能确定日期，需说明。

2. **来源可追溯**：信息来源必须写明可追溯来源。

3. **禁止编造**：只基于可验证信息。无数据时如实标注，不得用常识填充。

4. **固定输出格式**（每个维度5字段）：

```json
{
  "信息发生时间": "...",
  "信息来源": "...（至少1个核心来源）",
  "信息总结": "...",
  "对市场的影响": "...（对股价/合约价格的具体影响路径）",
  "这个信息能影响多久": "..."
}
```

5. **实时搜索优先**：每个子问题必须先执行实时搜索。

6. **数据留白原则**：无数据时标注"未在窗口期内检索到对应数据"。

7. **一致性原则**：矛盾数据标注矛盾并优先采信最新时间戳。

#### 各维度搜索指引

**维度1 - 货币政策**：
- 搜索：`Fed interest rate decision`、`FOMC minutes`、`DXY`、`real yield TIPS`
- 推荐：CME FedWatch、FRED
- 窗口：30天

**维度2 - 经济周期**：
- 搜索：`US GDP latest`、`nonfarm payrolls`、`CPI release`、`ISM PMI`、`yield curve 2s10s`
- 推荐：BLS.gov、BEA.gov
- 窗口：30天（月度数据）

**维度3 - 财政政策**：
- 搜索：`federal budget deficit`、`debt ceiling`、`tax policy US`、`fiscal stimulus`
- 推荐：CBO.gov、Treasury.gov
- 窗口：30天

**维度4 - 行业景气度**：
- 搜索：品种所属行业 + `sector performance`、`sector rotation`、`industry outlook 2026`
- GICS行业映射：NVDA→半导体、AMZN→消费/云、JPM→金融、LLY→医药
- 窗口：7天

**维度5 - 比较优势**：
- 搜索：品种公司名 + `market share`、`competitive moat`、`industry leader`
- 关注：管理层讨论（MD&A）中的竞争定位描述
- 窗口：30天

**维度6 - 盈利能力**：
- 搜索：品种公司名 + `quarterly earnings`、`revenue growth`、`profit margin`、`ROE`、`free cash flow`
- 推荐：SEC EDGAR（10-Q/10-K）、Yahoo Finance Statistics
- 窗口：30天（覆盖最新财报季）

**维度7 - 估值安全边际**：
- 搜索：品种公司名 + `PE ratio`、`forward PE`、`PEG ratio`、`valuation vs peers`
- 关注：当前PE vs 5年均值分位、vs行业中位数偏离
- 窗口：24小时

**维度8 - 盈利预期**：
- 搜索：品种公司名 + `earnings surprise`、`guidance raised lowered`、`analyst upgrade downgrade`
- 关注：最近一季beat/miss幅度、下一季guidance方向
- 窗口：30天

**维度9 - 合约资金流**（Binance API 优先）：
- 数据源：**Binance API `open_interest` + `funding_rate`**
- OI+价格组合信号：
  - 涨+OI增 → 多头趋势确认
  - 涨+OI减 → 空头平仓推动，脆弱
  - 跌+OI增 → 空头趋势确认
  - 跌+OI减 → 多头平仓，可能见底
- 资金费率：正=多头拥挤；负=空头拥挤
- 窗口：4小时
- ⚠️ 永续合约特有维度，O权重应高于平均

**维度10 - 机构动向**：
- 搜索：品种代码 + `ETF inflows outflows`、`13F filing`、`insider selling buying`、`short interest percent`
- 关注：对应ETF（如NVDA→SOXX/SMH）资金流、知名基金持仓变动
- 窗口：7天

**维度11 - 市场情绪**：
- 搜索：`VIX index`、`equity put call ratio`、`AAII bull bear spread`、`CNN fear greed`
- 关注：VIX>25=恐慌（反向做多信号）；<12=自满（警惕）
- 窗口：4小时

**维度12 - 技术结构**（Binance API 优先）：
- 数据源：**Binance API K线** → 计算MA20/MA50/MA200/布林带/ATR
- 关注：RS vs SPY（相对强弱）、均线排列、量价关系
- 注意：永续K线与美股现货高度同步但24/7有差异
- 窗口：24小时

**维度13 - 公司治理**：
- 搜索：品种公司名 + `board independence`、`share buyback program`、`dividend increase`、`CEO compensation`
- 关注：回购计划规模占市值比、分红政策稳定性
- 窗口：30天

**维度14 - 事件驱动**：
- 搜索：品种公司名 + `next earnings date`、`product launch`、`FDA decision`、`M&A rumor`、`SEC filing`
- 提前标注未来事件日历
- 窗口：7天

---

### 第二步：维度评分

在每个维度根据获取到的数据和信息，分析对品种永续合约价格的影响程度打分：
- **最大利多**：1分
- **中性**：0.5分
- **最大利空**：0分

输出：**14维度打分表**

| 维度 | 得分 | 评分理由 |
|------|------|----------|
| 1 货币政策 | x.xx | ... |
| ... | ... | ... |
| 14 事件驱动 | x.xx | ... |

---

### 第三步：权重评估

权重调整原则：
- **FOMC周** → D1↑
- **财报季/即将ER** → D6↑ D8↑ D14↑
- **VIX>25或<12** → D11↑
- **OI异常/费率极端** → D9↑
- **行业轮动期** → D4↑
- **ETF大幅流入/流出** → D10↑

所有维度权重之和 = 1.0

---

### 第四步：加权平均评估

$$综合得分 = \sum_{i=1}^{14} (得分_i \times 权重_i)$$

**多空研判**：

| 区间 | 判定 |
|------|------|
| 0.00-0.30 | 🔴 强利空 |
| 0.30-0.45 | 🟠 偏空 |
| 0.45-0.55 | 🟡 中性观望 |
| 0.55-0.70 | 🟢 偏多 |
| 0.70-1.00 | 🔵 强利多 |

---

### 第五步：操作方案生成

#### 5.1 方向判断 + 多维度交叉确认（至少2条满足才可开仓）：
- 合约资金流（D9）信号一致
- 技术结构（D12）支持该方向
- 盈利预期（D8）趋势支持

#### 5.2 仓位建议
| 综合得分偏离0.5 | 建议杠杆 |
|----------------|---------|
| 0.05-0.10 | 0.3× |
| 0.10-0.20 | 0.3× |
| >0.20 | 0.5× |

#### 5.3 约束条件
- **中性区间**（0.45-0.55）：观望，不主动开仓
- **费率保护**：>0.1%→做多降仓；<-0.1%→做空降仓
- **OI过热**：30日极值→标注警告
- **事件前**：重大事件（ER/FOMC/FDA）→标注风险管理提示

---

## 输出产物

1. 14维度信息采集报告（每个维度5字段格式）
2. 14维度打分表（含评分和理由）
3. 14维度权重表（含权重和调整理由）
4. 综合评估表（加权得分+多空研判）
5. 操作方案（方向/仓位/入场/止损/止盈）
6. 保存为 `.md` 文件至 `./StockAnalysis/reports/{品种名}/`

## 注意事项
- 所有搜索必须实时执行，严禁使用过时缓存
- 无数据维度默认0.5（中性），权重降低
- 分析中发现数据矛盾必须标注并说明取舍依据
- 报告末尾必须附免责声明
- **窗口期差异化**：实时4h / 日度24h / 周度7d / 月度30d
