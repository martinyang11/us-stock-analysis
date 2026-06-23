#!/usr/bin/env python3
"""
SA 每日评分脚本 (2026-06-23)
计算 56 个品种的 14 维度基本面评分 + 24 技术指标
"""
import sys, os, logging, json, csv, io
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
log = logging.getLogger("SA")

# 运行时日期（每次脚本执行时确定，--date 可覆盖）
_RUN_DATE = datetime.now(timezone.utc)
if '--date' in sys.argv:
    idx = sys.argv.index('--date')
    if idx + 1 < len(sys.argv):
        date_arg = sys.argv[idx + 1]
        _RUN_DATE = datetime.strptime(date_arg, '%Y%m%d').replace(tzinfo=timezone.utc)
_RUN_DATE_STR = _RUN_DATE.strftime('%Y-%m-%d')
_RUN_DATE_COMPACT = _RUN_DATE.strftime('%Y%m%d')
_RUN_MONTH = _RUN_DATE.month

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf

from skills.StockAnalysis.scripts.gtrade_data import (
    GtradeDataProvider, compute_technical_scores, get_tradfi_pairs
)

# ============================================================
# 1. 获取品种列表
# ============================================================
pairs = get_tradfi_pairs()
VARIETIES = []
for i, p in enumerate(pairs):
    VARIETIES.append({
        'variety_id': i,  # 顺序内部ID (0-55), 非gTrade pairIndex
        'pair_index': p['pairIndex'],  # gTrade 原始 pairIndex
        'variety_name': p['name'],
        'category': p.get('group', 'stocks'),
        'group': p.get('group', ''),
    })
log.info(f"获取到 {len(VARIETIES)} 个品种")

# Sort by variety_id
VARIETIES.sort(key=lambda x: x['variety_id'])

# ============================================================
# 2. 计算 24 维技术指标
# ============================================================
def compute_tech_for_variety(name: str) -> list:
    """用 yfinance 计算 24 维技术指标"""
    try:
        ticker_map = {
            'XAU': 'GC=F', 'XAG': 'SI=F', 'WTI': 'CL=F', 'XPT': 'PL=F',
            'XPD': 'PA=F', 'HG': 'HG=F', 'NATGAS': 'NG=F', 'BRENT': 'BZ=F',
            'SPX500': '^GSPC', 'NAS100': '^NDX', 'USA30': '^DJI',
            'URNM': 'URNM', 'URA': 'URA', 'GDX': 'GDX', 'WPM': 'WPM',
            'CRCL': 'CRCL', 'SBET': 'SBET',
        }
        ticker = ticker_map.get(name, name)

        df = yf.download(ticker, period='1y', interval='1d', progress=False, auto_adjust=True)
        if df is None or len(df) < 20:
            log.warning(f"{name}: yfinance 数据不足 ({len(df) if df is not None else 0} 行), 返回中性")
            return [0.5] * 24

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten MultiIndex: ('Close', 'AAPL') -> 'close'
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        scores = compute_technical_scores(df)
        return scores
    except Exception as e:
        log.warning(f"{name}: 技术指标计算失败 ({e})")
        return [0.5] * 24

log.info("开始计算 24 维技术指标...")
tech_scores = {}
for i, v in enumerate(VARIETIES):
    name = v['variety_name']
    if (i + 1) % 10 == 0:
        log.info(f"  进度: {i+1}/{len(VARIETIES)}")
    tech_scores[name] = compute_tech_for_variety(name)
log.info(f"技术指标计算完成: {len(tech_scores)} 个品种")

# ============================================================
# 3. gTrade 数据 (D9: 合约资金流)
# ============================================================
log.info("获取 gTrade 数据...")
d9_scores = {}
market_spreads = {}
try:
    with GtradeDataProvider(use_ws=False) as provider:
        for v in VARIETIES:
            name = v['variety_name']
            try:
                spread = provider.get_spread(name)
                vol = provider.get_24h_volume(name)
                market_spreads[name] = {'spread': spread, 'volume': vol}
            except:
                market_spreads[name] = {'spread': None, 'volume': None}
except Exception as e:
    log.warning(f"gTrade 数据获取失败: {e}")

# ============================================================
# 4. 维度评分
# ============================================================
# 基于 2026-06-23 研究数据的宏观评分

# D1: 货币政策 - FOMC后加息概率升至89%, Warsh鹰派基调确认
# dot plot 中位数 3.8%(+40bp), 2Y收益率4.23%(16月高), 10Y 4.51%
# Goldman: 加息是真实风险, HSBC: USD将持续走强
# 6/25 PCE通胀数据是关键催化剂, 市场如惊弓之鸟
# 基础分 0.25 (紧缩强化+加息概率逼近90%)
D1_BASE = 0.25

# D2: 经济周期 - GDP 2.2%, ISM 54.0, NFP +172K, 失业率 4.3%
# CPI 4.2% YoY, 但油价-4.8%(伊朗和平)缓解通胀压力, 消费者受益
# 核心CPI 2.9%仍高, PPI 6.5%, 但能源供应改善→滞胀风险边际下降
# 基础分 0.50 (扩张+能源成本下降, 但加息逆风)
D2_BASE = 0.50

# D3: 财政政策 - 国债 $38.5T→$41.1T上限, 赤字 ~$2T/年, 利息 >$1T/年
# Greenspan(前Fed主席)去世, 财政改革政治僵局持续
# SpaceX $20B债券发行引发债务市场关注
# 基础分 0.40 (财政恶化持续, 无改革迹象)
D3_BASE = 0.40

# D11: 市场情绪 - VIX飙18%至20.54, 全球抛售: Nasdaq期货-2.55%, Nikkei -3.55%, KOSPI -10%
# GOOGL -7%(AI人才流失: AlphaFold诺贝尔得主+Gemini联创双离职)
# SpaceX SPCX -16%($20B债券), Mag7再承压, AI capex ROI质疑升温
# F&G指数约30-33(Fear), 全球risk-off, 6/25 PCE是下一个雷
# 基础分 0.33 (恐惧加剧, AI恐慌+加息恐惧双重打击)
D11_BASE = 0.33

# 类别调整系数: (D1_mult, D2_mult, D3_mult, D11_mult)
CATEGORY_ADJUST = {
    # 科技/成长: 高利率最受伤, 经济扩张利好
    'tech':       (0.85, 1.10, 1.00, 1.05),
    'semi':       (0.80, 1.15, 1.00, 1.00),
    # 金融/支付 - V/MA接近52周低点, 但收入+15%, 稳定币平台叙事: 高利率利好净息差
    'financial':  (1.25, 1.10, 1.00, 1.00),
    # 消费 - NKE/SBUX转型进行中, KO/WMT/MCD防御性稳定: 经济好利好, 利率中性
    'consumer':   (1.00, 1.15, 1.00, 1.00),
    # 工业/国防 - 伊朗停火利空, 但FY27国防预算$1.5T: 经济敏感
    'industrial': (0.95, 1.20, 1.05, 1.00),
    # 加密 - BTC $64K横盘, 美伊和平利好, 但加息预期压制相关: 高风险, 对利率敏感, 情绪敏感
    'crypto':     (0.90, 0.95, 1.00, 1.20),
    # 加密 - BTC $64K横盘, 美伊和平利好, 但加息预期压制原生
    'crypto_native': (0.95, 0.95, 1.00, 1.15),
    # 商品: 通胀高利好, 美元强利空
    'commodity':  (1.10, 1.15, 1.00, 0.95),
    # 指数/ETF: 综合市场
    'index':      (0.95, 1.05, 1.00, 1.00),
    # 医药
    'pharma':     (0.95, 0.95, 1.00, 0.95),
    # 中国科技 - 五角大楼黑名单, AI收入52%首次过半, Apollo Go欧洲扩张
    'china_tech': (0.90, 0.90, 0.95, 0.90),
    # 矿业ETF
    'mining':     (1.05, 1.15, 1.00, 1.00),
    # 铀/能源
    'energy':     (1.10, 1.10, 1.00, 0.95),
    # 其他
    'other':      (1.00, 1.00, 1.00, 1.00),
}

# 品种 → 分类映射
VARIETY_CATEGORY = {
    'BTC': 'crypto_native', 'ETH': 'crypto_native',
    'AAPL': 'tech', 'GOOGL': 'tech', 'AMZN': 'tech', 'MSFT': 'tech',
    'TSLA': 'tech', 'META': 'tech', 'NFLX': 'tech',
    'NVDA': 'semi', 'AMD': 'semi', 'INTC': 'semi',
    'V': 'financial', 'MA': 'financial', 'PYPL': 'financial',
    'COIN': 'crypto', 'HOOD': 'crypto', 'MSTR': 'crypto', 'MARA': 'crypto', 'RIOT': 'crypto',
    'KO': 'consumer', 'DIS': 'consumer', 'NKE': 'consumer',
    'SBUX': 'consumer', 'WMT': 'consumer', 'MCD': 'consumer',
    'GME': 'other', 'SNAP': 'tech', 'ABNB': 'tech', 'ROKU': 'tech',
    'BA': 'industrial', 'LMT': 'industrial',
    'PFE': 'pharma',
    'BIDU': 'china_tech',
    'PLTR': 'tech', 'CRCL': 'other', 'SBET': 'other',
    'SPY': 'index', 'QQQ': 'index', 'IWM': 'index', 'DIA': 'index',
    'SPX500': 'index', 'NAS100': 'index', 'USA30': 'index',
    'XAU': 'commodity', 'XAG': 'commodity', 'WTI': 'energy',
    'XPT': 'commodity', 'XPD': 'commodity', 'HG': 'commodity',
    'NATGAS': 'energy', 'BRENT': 'energy',
    'GDX': 'mining', 'URA': 'energy', 'WPM': 'mining', 'URNM': 'energy',
}

# ============================================================
# 5. 行业评分 (D4)
# ============================================================
# 基于 2026-06-23 行业研究数据
D4_INDUSTRY = {
    # 半导体 - AVGO AI指引逊预期砸$1.4T但已反弹, HBM供不应求: SOX从-5.7%反弹, AVGO财测逊预期砸$1.4T但AI需求完整, 存储芯片(HBM)供不应求
    'semi': 0.52,  # SOX创新高, MU HBM售罄, 存储芯片最强
    # 科技巨头 - Mag7月内蒸发$2T但恢复中, SpaceX IPO提振, 板块轮动: Mag7月内蒸发$2T但恢复中, 板块轮动到价值/小盘, AI capex $650B仍强劲
    'tech': 0.45,  # GOOGL AI人才流失, SpaceX -16%, 纳指-1.32%, AI ROI质疑
    # 加密 - BTC $64K横盘, 美伊和平利好, 但加息预期压制相关: BTC $64K, 美伊和平协议利好, 但加息预期压制, CLARITY法案停滞
    'crypto': 0.52,
    'crypto_native': 0.54,
    # 金融/支付 - V/MA接近52周低点, 但收入+15%, 稳定币平台叙事: V/MA接近52周低点但基本面强劲(收入+15%), 加息利好净息差
    'financial': 0.54,  # 高利率利好, 但risk-off情绪蔓延
    # 消费 - NKE/SBUX转型进行中, KO/WMT/MCD防御性稳定: 防御性抗跌, Nike/SBUX复苏进行中, KO/WMT/MCD相对稳定
    'consumer': 0.55,  # 油价-4.8%利好消费, 防御性轮动
    # 工业/国防 - 伊朗停火利空, 但FY27国防预算$1.5T: 伊朗停火利空国防, LMT -23%(3个月), 但FY27国防预算$1.5T创纪录
    'industrial': 0.48,  # SpaceX $20B债压垮太空板块, 拖累工业情绪
    # 医药: 防御性, 利率敏感度低
    'pharma': 0.52,
    # 中国科技 - 五角大楼黑名单, AI收入52%首次过半, Apollo Go欧洲扩张: 五角大楼将BIDU/阿里/BYD列入军方企业清单, 恒生科技承压
    'china_tech': 0.36,  # 全球risk-off+五角大楼黑名单, KOSPI -10%蔓延
    # 指数/ETF: SPY从4月低点+19%, DIA接近ATH, IWM领涨(板块轮动)
    'index': 0.48,  # 全球抛售, Nasdaq期货-2.55%, 亚太暴跌
    # 商品: 黄金受美元走强压制, 油价受伊朗影响波动($86-88), 铜需求稳健
    'commodity': 0.48,  # 黄金-2%至$4105, 美元强势压制
    'energy': 0.42,  # 油价-4.8%至$73.64, 伊朗和平+供给恢复
    'mining': 0.54,
    # 其他
    'other': 0.48,
}

# ============================================================
# 6. 公司特定评分 (D5-D8, D10, D13-D14)
# 基于搜索研究 + 前次评分调整
# ============================================================
# 阅读前次评分作为基线（自动找最近的上一个评分文件）
prev_scores = {}
scores_glob = sorted(Path('skills/SANN/data/daily_scores').glob('scores_*.csv'))
# 排除今天的文件，取最新的一个
prev_files = [f for f in scores_glob if f.stem != f'scores_{_RUN_DATE_COMPACT}']
if prev_files:
    prev_file = prev_files[-1]  # 按文件名排序，最后一个是最新的
    with open(prev_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            prev_scores[row['variety_name']] = row
    log.info(f"加载前次评分: {len(prev_scores)} 个品种 (来源: {prev_file.name})")
else:
    log.info(f"无前次评分文件可加载（首次运行？）")

def get_prev(name, dim, default=0.50):
    """获取前次评分, 带默认值"""
    if name in prev_scores:
        return float(prev_scores[name].get(dim, default))
    return default

# 公司特定评分 (基于 2026/6 研究)
# 格式: {品种名: {dim5, dim6, dim7, dim8, dim10, dim13, dim14}}
#
# 关键研究结论 (2026-06-23):
# - GOOGL: -7%! AlphaFold诺贝尔得主Jumper+Gemini联创Shazeer双双离职, AI人才流失严重
# - NVDA: AI需求完整, DGX选中INTC Xeon 6, BofA正面, 但AI ROI质疑升温
# - AMD: MI450延迟, DC收入$5.8B, YTD +118%, SOX创新高受益
# - MU: 6/24财报, HBM售罄2026全年, 分析师极度看涨(目标$1,200-1,550)
# - BTC: $62-64K横盘, 加息预期89%压制 vs 伊朗和平利好
# - SpaceX SPCX: -16%! $20B首次债券发行, $400-600B市值蒸发
# - V/MA: 52周低点, 收入+15%, 加息利好, 防御性轮动受益
# - WTI: $73.64(-4.8%), 伊朗60天和平路线图+油轮恢复装载, 供给改善
# - XAU: $4,105(-2%), 美元强势+加息预期压制, 但地缘不确定性支撑
# - BIDU: 五角大楼黑名单, AI收入占52%, Apollo Go欧洲扩张, KOSPI -10%蔓延
# - 6/25 PCE通胀数据: 下一个关键催化剂, 决定加息预期走向

COMPANY_SCORES = {
    # 加密 - BTC $64K横盘, 美伊和平利好, 但加息预期压制
    'BTC':  {'dim5': 0.78, 'dim6': 0.55, 'dim7': 0.50, 'dim8': 0.55, 'dim10': 0.65, 'dim13': 0.50, 'dim14': 0.62},
    'ETH':  {'dim5': 0.72, 'dim6': 0.55, 'dim7': 0.50, 'dim8': 0.55, 'dim10': 0.60, 'dim13': 0.50, 'dim14': 0.58},
    # 科技巨头 - Mag7月内蒸发$2T但恢复中, SpaceX IPO提振, 板块轮动
    'NVDA': {'dim5': 0.85, 'dim6': 0.75, 'dim7': 0.32, 'dim8': 0.65, 'dim10': 0.62, 'dim13': 0.65, 'dim14': 0.55},
    'AMD':  {'dim5': 0.65, 'dim6': 0.58, 'dim7': 0.38, 'dim8': 0.58, 'dim10': 0.58, 'dim13': 0.55, 'dim14': 0.52},
    'INTC': {'dim5': 0.32, 'dim6': 0.35, 'dim7': 0.62, 'dim8': 0.32, 'dim10': 0.38, 'dim13': 0.40, 'dim14': 0.38},
    'AAPL': {'dim5': 0.78, 'dim6': 0.68, 'dim7': 0.42, 'dim8': 0.55, 'dim10': 0.58, 'dim13': 0.68, 'dim14': 0.50},
    'MSFT': {'dim5': 0.80, 'dim6': 0.72, 'dim7': 0.40, 'dim8': 0.62, 'dim10': 0.60, 'dim13': 0.65, 'dim14': 0.52},
    'GOOGL':{'dim5': 0.70, 'dim6': 0.58, 'dim7': 0.50, 'dim8': 0.48, 'dim10': 0.52, 'dim13': 0.50, 'dim14': 0.48},  # AI人才流失: Jumper+Shazeer双离职
    'AMZN': {'dim5': 0.75, 'dim6': 0.60, 'dim7': 0.50, 'dim8': 0.55, 'dim10': 0.55, 'dim13': 0.55, 'dim14': 0.48},  # AI capex ROI质疑
    'META': {'dim5': 0.70, 'dim6': 0.65, 'dim7': 0.45, 'dim8': 0.58, 'dim10': 0.52, 'dim13': 0.50, 'dim14': 0.48},  # AI spending压力
    'TSLA': {'dim5': 0.68, 'dim6': 0.52, 'dim7': 0.35, 'dim8': 0.50, 'dim10': 0.52, 'dim13': 0.45, 'dim14': 0.55},
    'NFLX': {'dim5': 0.70, 'dim6': 0.65, 'dim7': 0.45, 'dim8': 0.55, 'dim10': 0.55, 'dim13': 0.55, 'dim14': 0.48},
    # 半导体 - AVGO AI指引逊预期砸$1.4T但已反弹, HBM供不应求
    'SNAP': {'dim5': 0.40, 'dim6': 0.35, 'dim7': 0.55, 'dim8': 0.35, 'dim10': 0.35, 'dim13': 0.40, 'dim14': 0.38},
    'PLTR': {'dim5': 0.65, 'dim6': 0.55, 'dim7': 0.35, 'dim8': 0.58, 'dim10': 0.55, 'dim13': 0.50, 'dim14': 0.52},
    # 金融/支付 - V/MA接近52周低点, 但收入+15%, 稳定币平台叙事
    'V':   {'dim5': 0.78, 'dim6': 0.72, 'dim7': 0.50, 'dim8': 0.55, 'dim10': 0.58, 'dim13': 0.62, 'dim14': 0.48},
    'MA':  {'dim5': 0.78, 'dim6': 0.72, 'dim7': 0.50, 'dim8': 0.55, 'dim10': 0.58, 'dim13': 0.62, 'dim14': 0.48},
    'PYPL':{'dim5': 0.52, 'dim6': 0.50, 'dim7': 0.55, 'dim8': 0.45, 'dim10': 0.48, 'dim13': 0.50, 'dim14': 0.48},
    # 加密 - BTC $64K横盘, 美伊和平利好, 但加息预期压制相关
    'COIN':{'dim5': 0.65, 'dim6': 0.55, 'dim7': 0.42, 'dim8': 0.55, 'dim10': 0.58, 'dim13': 0.48, 'dim14': 0.55},
    'HOOD':{'dim5': 0.60, 'dim6': 0.52, 'dim7': 0.45, 'dim8': 0.52, 'dim10': 0.55, 'dim13': 0.45, 'dim14': 0.52},
    'MSTR':{'dim5': 0.55, 'dim6': 0.42, 'dim7': 0.38, 'dim8': 0.58, 'dim10': 0.62, 'dim13': 0.42, 'dim14': 0.55},
    'MARA':{'dim5': 0.42, 'dim6': 0.38, 'dim7': 0.48, 'dim8': 0.52, 'dim10': 0.55, 'dim13': 0.40, 'dim14': 0.50},
    'RIOT':{'dim5': 0.42, 'dim6': 0.38, 'dim7': 0.48, 'dim8': 0.52, 'dim10': 0.55, 'dim13': 0.40, 'dim14': 0.50},
    # 消费 - NKE/SBUX转型进行中, KO/WMT/MCD防御性稳定
    'KO':  {'dim5': 0.72, 'dim6': 0.65, 'dim7': 0.55, 'dim8': 0.48, 'dim10': 0.50, 'dim13': 0.60, 'dim14': 0.45},
    'DIS': {'dim5': 0.65, 'dim6': 0.52, 'dim7': 0.50, 'dim8': 0.52, 'dim10': 0.52, 'dim13': 0.50, 'dim14': 0.50},
    'NKE': {'dim5': 0.60, 'dim6': 0.48, 'dim7': 0.55, 'dim8': 0.45, 'dim10': 0.45, 'dim13': 0.50, 'dim14': 0.42},
    'SBUX':{'dim5': 0.62, 'dim6': 0.55, 'dim7': 0.52, 'dim8': 0.50, 'dim10': 0.48, 'dim13': 0.52, 'dim14': 0.45},
    'WMT': {'dim5': 0.75, 'dim6': 0.65, 'dim7': 0.52, 'dim8': 0.50, 'dim10': 0.52, 'dim13': 0.58, 'dim14': 0.45},
    'MCD': {'dim5': 0.70, 'dim6': 0.62, 'dim7': 0.52, 'dim8': 0.48, 'dim10': 0.50, 'dim13': 0.55, 'dim14': 0.45},
    # 工业/国防 - 伊朗停火利空, 但FY27国防预算$1.5T
    'BA':  {'dim5': 0.58, 'dim6': 0.42, 'dim7': 0.50, 'dim8': 0.50, 'dim10': 0.50, 'dim13': 0.45, 'dim14': 0.52},
    'LMT': {'dim5': 0.68, 'dim6': 0.60, 'dim7': 0.50, 'dim8': 0.52, 'dim10': 0.52, 'dim13': 0.52, 'dim14': 0.55},
    # 医药
    'PFE': {'dim5': 0.58, 'dim6': 0.48, 'dim7': 0.58, 'dim8': 0.45, 'dim10': 0.45, 'dim13': 0.52, 'dim14': 0.48},
    # 中国科技 - 五角大楼黑名单, AI收入52%首次过半, Apollo Go欧洲扩张
    'BIDU':{'dim5': 0.52, 'dim6': 0.48, 'dim7': 0.58, 'dim8': 0.42, 'dim10': 0.40, 'dim13': 0.40, 'dim14': 0.40},
    # 模因
    'GME': {'dim5': 0.25, 'dim6': 0.28, 'dim7': 0.35, 'dim8': 0.25, 'dim10': 0.40, 'dim13': 0.30, 'dim14': 0.42},
    # 互联网/流媒体 - ROKU受SpaceX IPO积极情绪提振
    'ABNB':{'dim5': 0.58, 'dim6': 0.55, 'dim7': 0.50, 'dim8': 0.52, 'dim10': 0.50, 'dim13': 0.50, 'dim14': 0.48},
    'ROKU':{'dim5': 0.48, 'dim6': 0.42, 'dim7': 0.52, 'dim8': 0.45, 'dim10': 0.45, 'dim13': 0.45, 'dim14': 0.45},
    # 其他
    'CRCL':{'dim5': 0.42, 'dim6': 0.35, 'dim7': 0.50, 'dim8': 0.40, 'dim10': 0.38, 'dim13': 0.40, 'dim14': 0.42},
    'SBET':{'dim5': 0.40, 'dim6': 0.32, 'dim7': 0.50, 'dim8': 0.38, 'dim10': 0.35, 'dim13': 0.38, 'dim14': 0.40},
}

# 指数/ETF: 继承成分股平均
INDEX_NAMES = {'SPY', 'QQQ', 'IWM', 'DIA', 'SPX500', 'NAS100', 'USA30'}
for name in INDEX_NAMES:
    if name not in COMPANY_SCORES:
        COMPANY_SCORES[name] = {
            'dim5': 0.65, 'dim6': 0.60, 'dim7': 0.50, 'dim8': 0.55,
            'dim10': 0.55, 'dim13': 0.55, 'dim14': 0.50
        }

# 商品: 无公司基本面, 中性
COMMODITY_NAMES = {'XAU', 'XAG', 'WTI', 'XPT', 'XPD', 'HG', 'NATGAS', 'BRENT'}
for name in COMMODITY_NAMES:
    if name not in COMPANY_SCORES:
        COMPANY_SCORES[name] = {
            'dim5': 0.55, 'dim6': 0.48, 'dim7': 0.50, 'dim8': 0.50,
            'dim10': 0.45, 'dim13': 0.50, 'dim14': 0.50
        }

# 矿业/铀 ETF
ETF_NAMES = {'GDX', 'URA', 'WPM', 'URNM'}
for name in ETF_NAMES:
    if name not in COMPANY_SCORES:
        COMPANY_SCORES[name] = {
            'dim5': 0.58, 'dim6': 0.52, 'dim7': 0.50, 'dim8': 0.52,
            'dim10': 0.55, 'dim13': 0.50, 'dim14': 0.50
        }

# ============================================================
# 7. D12 技术结构评分 (基于 24 维技术指标综合)
# ============================================================
def compute_d12(tech24: list) -> float:
    """从 24 维技术指标计算 D12 技术结构评分"""
    if not tech24 or len(tech24) < 24:
        return 0.50
    # T1-T3: 均线偏离 (多头=高), T4: 排列, T8: RSI, T10-T13: 动量
    # 综合均线+动量+RSI
    ma_score = (tech24[0] + tech24[1] + tech24[2]) / 3  # T1-T3
    alignment = tech24[3]  # T4
    rsi = tech24[7]  # T8
    momentum = (tech24[9] + tech24[10] + tech24[11] + tech24[12]) / 4  # T10-T13
    boll_pos = tech24[4]  # T5

    d12 = (ma_score * 0.25 + alignment * 0.15 + rsi * 0.20 + momentum * 0.25 + boll_pos * 0.15)
    return round(np.clip(d12, 0.0, 1.0), 4)

# ============================================================
# 8. D9 合约资金流 (基于 gTrade spread + volume)
# ============================================================
def compute_d9(name: str) -> float:
    """从 gTrade 数据计算 D9"""
    info = market_spreads.get(name, {})
    spread = info.get('spread')
    vol = info.get('volume')

    base = 0.50
    # Spread: 越低越好 (流动性好)
    if spread is not None:
        if spread < 0.0005:   # <5bps
            base += 0.10
        elif spread < 0.001:  # <10bps
            base += 0.05
        elif spread > 0.005:  # >50bps
            base -= 0.10

    # Volume: 越高越好
    if vol is not None and vol > 0:
        # 对数化处理
        log_vol = np.log10(vol)
        if log_vol > 6:
            base += 0.08
        elif log_vol > 5:
            base += 0.04
        elif log_vol < 3:
            base -= 0.08

    return round(np.clip(base, 0.0, 1.0), 4)

# ============================================================
# 9. 生成最终评分
# ============================================================
def generate_scores():
    """为所有品种生成完整评分"""
    rows = []
    today = _RUN_DATE_STR
    month = _RUN_MONTH

    for v in VARIETIES:
        name = v['variety_name']
        vid = v['variety_id']
        cat = VARIETY_CATEGORY.get(name, 'other')
        adj = CATEGORY_ADJUST.get(cat, CATEGORY_ADJUST['other'])

        # D1-D3, D11: 宏观 + 类别调整
        d1 = round(np.clip(D1_BASE * adj[0], 0.0, 1.0), 4)
        d2 = round(np.clip(D2_BASE * adj[1], 0.0, 1.0), 4)
        d3 = round(np.clip(D3_BASE * adj[2], 0.0, 1.0), 4)
        d4 = round(D4_INDUSTRY.get(cat, 0.50), 4)
        d11 = round(np.clip(D11_BASE * adj[3], 0.0, 1.0), 4)

        # D5-D8, D10, D13-D14: 公司特定
        cs = COMPANY_SCORES.get(name, {})
        d5 = round(cs.get('dim5', 0.50), 4)
        d6 = round(cs.get('dim6', 0.50), 4)
        d7 = round(cs.get('dim7', 0.50), 4)
        d8 = round(cs.get('dim8', 0.50), 4)
        d10 = round(cs.get('dim10', 0.50), 4)
        d13 = round(cs.get('dim13', 0.50), 4)
        d14 = round(cs.get('dim14', 0.50), 4)

        # D9: gTrade 合约资金流
        d9 = compute_d9(name)

        # D12: 技术结构
        t = tech_scores.get(name, [0.5]*24)
        d12 = compute_d12(t)

        row = [
            today, vid, name, month,
            d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14,
        ] + [round(x, 4) for x in t]

        rows.append(row)

    return rows

# ============================================================
# 10. 输出 CSV
# ============================================================
rows = generate_scores()

# Header
header = ['date', 'variety_id', 'variety_name', 'month',
          'dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8',
          'dim9', 'dim10', 'dim11', 'dim12', 'dim13', 'dim14'] + \
         [f'tech{i}' for i in range(1, 25)]

output_dir = Path('skills/SANN/data/daily_scores')
output_dir.mkdir(parents=True, exist_ok=True)
date_str = _RUN_DATE_COMPACT
output_file = output_dir / f'scores_{date_str}.csv'

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

log.info(f"评分文件已保存: {output_file}")
log.info(f"品种数: {len(rows)}")
log.info(f"维度列数: {len(header)}")

# 打印摘要
print("\n" + "=" * 70)
print(f"SA 评分摘要 ({_RUN_DATE_STR})")
print("=" * 70)
print(f"{'品种':<8} {'D1':>6} {'D2':>6} {'D3':>6} {'D4':>6} {'D5':>6} {'D6':>6} {'D7':>6} {'D8':>6} {'D9':>6} {'D10':>6} {'D11':>6} {'D12':>6} {'D13':>6} {'D14':>6} {'avg':>6}")
print("-" * 110)

# 按平均分排序
rows_with_avg = []
for r in rows:
    dims = r[4:18]  # dim1-dim14
    avg = sum(dims) / len(dims)
    rows_with_avg.append((avg, r))

rows_with_avg.sort(key=lambda x: x[0], reverse=True)

for avg, r in rows_with_avg[:10]:
    name = r[2]
    dims = r[4:18]
    dims_str = ' '.join(f'{d:>6.4f}' for d in dims)
    print(f"{name:<8} {dims_str} {avg:>6.4f}")

print("...")
for avg, r in rows_with_avg[-5:]:
    name = r[2]
    dims = r[4:18]
    dims_str = ' '.join(f'{d:>6.4f}' for d in dims)
    print(f"{name:<8} {dims_str} {avg:>6.4f}")

print(f"\n总品种数: {len(rows)}")
print(f"评分范围: {rows_with_avg[-1][0]:.4f} ~ {rows_with_avg[0][0]:.4f}")
print(f"平均分: {sum(a for a,_ in rows_with_avg)/len(rows_with_avg):.4f}")
