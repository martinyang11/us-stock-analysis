#!/usr/bin/env python3
"""
CA Auto-Scoring Engine v4.10
自动计算全品种38维CA评分（14基本面 + 24技术面），集成进CANN每日管线

数据源降级链（铁律）：AKShare → TQSDK → Sina nf_ → Web Search → None
严禁硬编码默认值（0.5等）。所有维度必须走降级链逐级获取，全部失败标记-1。

维度分层：
- 基本面(D1-D14)：宏观+产业基本面评分
  - 宏观层(D1-D6)：AKShare宏观API → 缓存 → Web Search → -1
  - 产业层(D7-D14)：AKShare K线+价差 → TQSDK实时K线 → Sina外盘 → Web Search → -1
- 技术面(T1-T24)：OHLCV K线数据计算，统一归入CA引擎
  - 数据源同产业层K线，走 AKShare → TQSDK 降级

变更日志:
  v4.10 (2026-06-10): D9缓存机制 — Agent预取基差数据，Python脚本只读
    - 新增 d9_basis_cache.json: 10个AKShare/TQSDK双失败的品种基差缓存
    - _load_d9_cache(): 首次调用加载JSON缓存到内存，后续调用直接读内存
    - _d9_web_search_fallback() 新增缓存优先: 缓存(confidence≥0.5) → Bing搜索 → None
    - 缓存miss品种(sc原油/ec欧线)回退到Bing搜索，搜索失败返回None(=D9=-1)
  v4.9 (2026-06-10): D9第三层降级 — TQSDK WebSocket永久不可用，新增Web Search兜底
    - _score_d9_basis() 新增 name 参数，第三层调用 _d9_web_search_fallback()
    - _d9_web_search_fallback(): Bing搜索"品种+基差+升贴水"，定量解析+定性判断双策略
    - 定性: "现货升水"=0.55 / "现货贴水"=0.45（方向信号，避免-1）
  v3.2 (2026-06-09): D8窗口5日→20日 — 解决D7↔D8共线性(r=0.8925)
    - D8供需四象限改用20日库存变化+20日价格变化，与D7(5日)形成短/中周期互补
    - 幅度微调系数下调（inv:2.0→0.8, price:1.5→0.6），适配更宽窗口
    - 最小数据要求从6点→21点
  v3.1 (2026-06-04): D7/D8修复 — 移除K线伪推断，接入真实仓单库存
    - D7 供应端: AKShare futures_inventory_em 仓单库存（5日变化率+20日分位），53/55品种覆盖
    - D8 需求端: 仓单库存×价格四象限联合判断（库存↓+价↑=需求旺盛等4种状态）
    - 原油sc/集运ec不在futures_inventory_em覆盖范围，D7/D8返回-1
    - 品种中文名映射 _INVENTORY_EM_NAME_MAP（55品种→53个覆盖）
  v3.0 (2026-06-03): CA统一38维评分（14基本面+24技术面）
    - 技术面归入CA：score_single_variety返回38维，CANN直接消费
    - CA综合分 = 38维算术均值（加权矩阵为未来规划，当前所有模型基于均值训练）
    - CSV输出从dim1-dim14扩展为dim1-dim38
  v4.8 (2026-06-05): D12历史回测季节性 — 硬编码SEASONALITY表替换为品种级K线回测
    - _compute_seasonality(): 取至少3年日线，按月度年收益率均值映射[0.35,0.65]
    - 结果缓存到 CANN/data/seasonality_cache.json，跨会话复用
  v4.7 (2026-06-05): D5从A股融资融券改为全市场OI变化率
    - 选取10个代表性品种（有色/黑色/能化/农产品各2-3个）计算近两日OI变化率中位数
    - OI增长=多头情绪，OI下降=避险情绪，至少5个品种有效才出分
  v4.6 (2026-06-05): _get_merged_news()增加fallback搜索 — 独立调用时自动搜Bing
    - _load_or_fetch_web_news(): 文件不存在/过期时fallback自己搜Bing，写入文件供复用
    - 确保管线预取和独立调用的D2/D3/D6信息源一致
  v4.5 (2026-06-05): D2/D3/D6新闻源升级 — CCTV替换为Web Search多平台聚合+SHMET补充
    - _get_merged_news(): Web Search(多平台)+SHMET快讯→合并文本(B方案)，跨源交叉验证
    - Web Search结果由管线预取到 CANN/data/web_news.txt，ca_scorer只读不搜
    - _fetch_shmet_news(): 保留AKShare SHMET期货快讯作为行业补充
    - 原 news_cctv 依赖完全移除
  v4.4 (2026-06-05): D10铜双市场融合 — LME铜(CAD)+COMEX铜(HG)等权平均
    - _compute_single_overseas 提取为独立函数，_score_d10_cross_market 支持列表输入
    - OVERSEAS_MAP 'cu' 从 'CAD' 改为 ['CAD', 'HG']，覆盖亚欧美全时区
    - 多市场等权平均，部分失败时自动降级到可用市场
  v4.3 (2026-06-05): D10外盘映射精细化 — 有色品种各自对应LME品种(CAD/AHD/ZSD/NID/SND/PBD)
    - 替换 HG(COMEX铜) 统一代理为LME一对一精确映射
    - LME代码通过futures_foreign_commodity_subscribe_exchange_symbol()获取
  v2.0 (2026-06-02): 全部14维度建立降级链，移除所有硬编码0.5
    - D1: AKShare M2+LPR → 缓存 → -1
    - D4: AKShare PMI+CPI → 缓存 → -1
    - D5: 代表性品种OI变化率中位数 → -1
    - D9: AKShare futures_spot_price_daily(基差) → TQSDK spread代理 → -1
    - D10: AKShare futures_foreign_hist(外盘历史) → 趋势对比+隔夜信号 → -1
    - D14: AKShare → TQSDK get_main_spread → -1
"""
import os, sys, json, time, warnings, logging, re
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'common'))
for p in [COMMON_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# monkey-patch for akshare (Python 3.13 compatible)
import pkgutil
class DummyImpImporter:
    def find_module(self, fullname, path=None): return None
pkgutil.ImpImporter = DummyImpImporter

from technical_indicators import compute_technical_scores

# 品种映射统一从 variety_list 导入
from skills.common.variety_list import get_variety_info, NUM_VARIETIES

# 构建品种信息列表（供 score_all_varieties 使用）
VARIETY_INFO = [get_variety_info(vid) for vid in range(NUM_VARIETIES)]

# 品种代码→AKShare现货品种代码映射（用于futures_spot_price_daily vars_list）
# 大部分品种代码与期货代码一致，少数特殊品种需要映射
SPOT_VARIETY_MAP = {
    'rb': 'RB', 'hc': 'HC', 'i': 'I', 'jm': 'JM', 'j': 'J',
    'sf': 'SF', 'sm': 'SM', 'cu': 'CU', 'al': 'AL', 'zn': 'ZN',
    'ni': 'NI', 'sn': 'SN', 'ao': 'AO', 'si': 'SI',
    'au': 'AU', 'ag': 'AG', 'sc': 'SC', 'fu': 'FU', 'lu': 'LU',
    'bu': 'BU', 'ru': 'RU', 'nr': 'NR', 'ta': 'TA', 'ma': 'MA',
    'sa': 'SA', 'fg': 'FG', 'pp': 'PP', 'l': 'L', 'v': 'V',
    'eg': 'EG', 'pf': 'PF', 'eb': 'EB', 'sp': 'SP', 'ur': 'UR',
    'a': 'A', 'b': 'B', 'm': 'M', 'y': 'Y', 'p': 'P',
    'oi': 'OI', 'rm': 'RM', 'c': 'C', 'cs': 'CS',
    'lh': 'LH', 'jd': 'JD', 'cf': 'CF', 'sr': 'SR',
    'ap': 'AP', 'cj': 'CJ', 'pk': 'PK', 'cy': 'CY',
    'lc': 'LC', 'ec': 'EC', 'pb': 'PB', 'ss': 'SS',
}

# 外盘映射（品种→AKShare futures_foreign_hist symbol）
# LME/SGX/NYMEX等代码通过futures_foreign_commodity_subscribe_exchange_symbol()获取
OVERSEAS_MAP = {
    'cu': ['CAD', 'HG'], 'al': 'AHD', 'zn': 'ZSD', 'ni': 'NID', 'sn': 'SND', 'pb': 'PBD',  # 有色→LME; cu双市场融合(LME+COMEX)
    'au': 'GC', 'ag': 'SI',                                                                 # 贵金属→COMEX
    'sc': 'CL', 'fu': 'CL', 'lu': 'CL', 'bu': 'CL',                                         # 能化上游→WTI（原油/燃油/沥青，产业链短）
    'ma': 'NG',                                                                             # 甲醇→天然气（海外甲醇以天然气为原料）
    'ta': None,                                                                              # PTA无合适外盘（PX在AKShare不可用，WTI链路太长致误判）
    'rb': None, 'hc': None, 'i': None, 'ss': None,                                           # 黑色系无直接外盘
}

# 外盘历史数据缓存（同一符号只拉一次）
_overseas_cache = {}

# ============================================================
# 缓存过期机制 — 所有模块级缓存TTL=1小时（支持管线+日常评估复用）
# ============================================================
_CACHE_HOUR = None  # 缓存所属小时（YYYYMMDDHH），None 表示未初始化

def _ensure_cache_fresh(force: bool = False):
    """确保所有模块级缓存对应当前小时。小时变更或force=True时重置全部缓存。
    
    TTL=1小时：平衡数据时效性与API调用开销，支持管线批量评分和日常单品种评估。
    
    受管理的缓存:
      - _overseas_cache (dict): 外盘历史数据
      - _shmet_news_cache: SHMET期货快讯
      - _merged_news_cache: 多源新闻聚合
      - _spread_akshare_cache (dict): AKShare价差
      - _ak_cache (dict): AKShare K线
      - _sina_kline_cache (dict): Sina K线 (v3.2)
    """
    global _overseas_cache, _shmet_news_cache, _merged_news_cache
    global _spread_akshare_cache, _ak_cache, _sina_kline_cache, _CACHE_HOUR
    hour_str = datetime.now().strftime('%Y%m%d%H')
    if force or _CACHE_HOUR != hour_str:
        _overseas_cache.clear()
        _shmet_news_cache = None
        _merged_news_cache = None
        _spread_akshare_cache.clear()
        _ak_cache.clear()
        _sina_kline_cache.clear()
        _CACHE_HOUR = hour_str
        if force:
            logger.info("缓存已强制刷新")
        elif _CACHE_HOUR is not None:
            logger.info(f"小时变更 → 缓存已重置 ({hour_str})")

# ============================================================
# D12 季节性缓存（按品种+月份，历史K线回测计算）
# ============================================================
_seasonality_cache = {}
_seasonality_cache_loaded = False
_SEASONALITY_CACHE_FILE = None

def _load_seasonality_cache():
    global _seasonality_cache_loaded
    if _seasonality_cache_loaded:
        return
    global _SEASONALITY_CACHE_FILE
    if _SEASONALITY_CACHE_FILE is None:
        _SEASONALITY_CACHE_FILE = os.path.join(SCRIPT_DIR, '..', 'data', 'seasonality_cache.json')
    try:
        if os.path.exists(_SEASONALITY_CACHE_FILE):
            with open(_SEASONALITY_CACHE_FILE, 'r') as f:
                raw = json.load(f)
                _seasonality_cache = {tuple(k.split('|')): v for k, v in raw.items()}
    except Exception:
        pass
    _seasonality_cache_loaded = True

def _save_seasonality_cache():
    global _SEASONALITY_CACHE_FILE
    if _SEASONALITY_CACHE_FILE is None:
        return
    try:
        os.makedirs(os.path.dirname(_SEASONALITY_CACHE_FILE), exist_ok=True)
        raw = {'|'.join([str(x) for x in k]): v for k, v in _seasonality_cache.items()}
        with open(_SEASONALITY_CACHE_FILE, 'w') as f:
            json.dump(raw, f)
    except Exception:
        pass

def _compute_seasonality(code: str, month: int) -> float:
    """基于历史K线回测计算品种月度季节性
    
    取至少3年日线数据，计算目标月份各年度的月收益率均值，
    线性映射到[0.35, 0.65]。结果缓存到磁盘。
    """
    import pandas as pd
    key = (code.upper(), month)
    if key in _seasonality_cache:
        return _seasonality_cache[key]
    
    try:
        import akshare as ak
        ak_code = code.upper() + '0'
        df = ak.futures_zh_daily_sina(symbol=ak_code)
        if df is None or len(df) < 252 * 3:
            _seasonality_cache[key] = 0.5
            return 0.5
        
        if 'datetime' in df.columns:
            df['date_col'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['date_col'] = pd.to_datetime(df['date'])
        else:
            _seasonality_cache[key] = 0.5
            return 0.5
        
        df = df.sort_values('date_col')
        df['_month'] = df['date_col'].dt.month
        df['_year'] = df['date_col'].dt.year
        df['_ret'] = df['close'].pct_change()
        
        month_data = df[df['_month'] == month]
        yearly_returns = []
        for yr, grp in month_data.groupby('_year'):
            if len(grp) >= 10:
                first_close = grp['close'].iloc[0]
                last_close = grp['close'].iloc[-1]
                if first_close > 0:
                    yearly_returns.append((last_close - first_close) / first_close)
        
        if len(yearly_returns) >= 3:
            avg_return = float(np.mean(yearly_returns))
            score = round(np.clip(0.5 + avg_return * 3, 0.35, 0.65), 4)
        else:
            score = 0.5
        
        _seasonality_cache[key] = score
        return score
    except Exception as e:
        logger.warning(f"季节回测失败({code}, m{month}): {e}")
        _seasonality_cache[key] = 0.5
        return 0.5


# ============================================================
# D1-D6: 宏观层 — AKShare API → 缓存 → -1
# ============================================================
def _score_d1_monetary() -> Tuple[float, str]:
    """D1 货币政策: AKShare M2+LPR → -1
    
    评分逻辑: M2增速适中+LPR稳定=偏宽松(0.5-0.6)，M2过高=宽松(>0.6)，M2过低=紧缩(<0.5)
    降级链: AKShare(macro_china_money_supply + macro_china_lpr) → None(-1)
    """
    try:
        import akshare as ak
        logging.getLogger('akshare').setLevel(logging.WARNING)
        
        # M2最新数据（iloc[0]是最新，数据为倒序）
        df_m2 = ak.macro_china_money_supply()
        latest_m2 = df_m2.iloc[0]
        m2_yoy = float(latest_m2['货币和准货币(M2)-同比增长'])
        
        # LPR（iloc[-1]是最新数据）
        df_lpr = ak.macro_china_lpr()
        latest_lpr = df_lpr.iloc[-1]
        lpr1y = float(latest_lpr['LPR1Y']) if latest_lpr['LPR1Y'] == latest_lpr['LPR1Y'] else 3.0
        
        # 评分: M2增速 8-12%正常, LPR 3-4%正常
        if m2_yoy > 15:
            m2_score = 0.75  # 极度宽松
        elif m2_yoy > 12:
            m2_score = 0.65
        elif m2_yoy >= 8:
            m2_score = 0.55  # 适度宽松
        elif m2_yoy >= 6:
            m2_score = 0.45
        else:
            m2_score = 0.35  # 紧缩
        
        if lpr1y <= 3.0:
            lpr_score = 0.65
        elif lpr1y <= 3.5:
            lpr_score = 0.55
        elif lpr1y <= 4.0:
            lpr_score = 0.45
        else:
            lpr_score = 0.35
        
        score = round(0.6 * m2_score + 0.4 * lpr_score, 4)
        source = f'AKShare:M2_yoy={m2_yoy}%,LPR1Y={lpr1y}%'
        return score, source
    except Exception as e:
        logger.warning(f"D1 AKShare失败: {e}")
        return -1.0, 'unavailable'


_shmet_news_cache = None  # SHMET期货快讯缓存（仍走AKShare）
_merged_news_cache = None  # 多源合并文本缓存


def _fetch_shmet_news() -> Tuple[str, str]:
    """从AKShare SHMET获取当日期货快讯（保留，补充Web Search）"""
    global _shmet_news_cache
    if _shmet_news_cache is not None:
        return _shmet_news_cache
    try:
        import akshare as ak
        df = ak.futures_news_shmet()
        if df is not None and len(df) > 0:
            contents = ' '.join(str(c) for c in df['内容'].tolist())
            _shmet_news_cache = (contents, f'SHMET({len(df)}条)')
            return _shmet_news_cache
    except Exception as e:
        logger.warning(f"futures_news_shmet失败: {e}")
    _shmet_news_cache = ('', 'unavailable')
    return _shmet_news_cache


def _get_merged_news() -> Tuple[str, str]:
    """多源新闻聚合: Web Search(多平台) + SHMET快讯 → 统一文本
    
    合并策略(B): 所有源文本拼接后统一打分，实现跨源交叉验证。
    优先读取管线预取的 web_news.txt（TTL=1小时），
    文件不存在/过期时 fallback 自己搜索Bing，确保独立调用也能拿到完整新闻。
    """
    global _merged_news_cache
    if _merged_news_cache is not None:
        return _merged_news_cache
    
    texts = []
    sources = []
    
    # 1. Web Search 新闻（管线预取或fallback自搜）
    web_file = os.path.join(SCRIPT_DIR, '..', 'data', 'web_news.txt')
    web_text = _load_or_fetch_web_news(web_file)
    if web_text:
        texts.append(web_text)
        sources.append(f'WebSearch({len(web_text)}字)')
    
    # 2. SHMET期货快讯（行业补充）
    shmet_text, shmet_src = _fetch_shmet_news()
    if shmet_text:
        texts.append(shmet_text)
        sources.append(shmet_src)
    
    merged = ' '.join(texts) if texts else ''
    source_str = '+'.join(sources) if sources else 'unavailable'
    _merged_news_cache = (merged, source_str)
    return _merged_news_cache


def _load_or_fetch_web_news(web_file: str) -> str:
    """读取管线预取的web_news.txt，过期/不存在时fallback自己搜Bing
    
    TTL=1小时，与管线 prefetch_web_news 一致。
    写入文件供后续调用复用。
    """
    # 缓存有效
    if os.path.exists(web_file):
        mtime = os.path.getmtime(web_file)
        size = os.path.getsize(web_file)
        if time.time() - mtime < 3600 and size > 100:
            try:
                with open(web_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception:
                pass
    
    # Fallback: 自己搜Bing
    logger.info("web_news.txt 不存在或已过期，fallback搜索Bing...")
    try:
        queries = [
            '"期货" 行情 分析 铜 原油 螺纹钢',
            '国内期货市场 有色金属 黑色系 能化 2026',
            '财政政策 专项债 赤字 基建投资 大宗商品 2026',
        ]
        all_snippets = []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        for q in queries:
            try:
                resp = requests.get('https://www.bing.com/search',
                                   params={'q': q, 'count': 10},
                                   headers=headers, timeout=15)
                if resp.status_code == 200:
                    blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)
                    for blk in blocks:
                        texts_in_block = re.findall(r'<(?:h2|p|a)[^>]*>(.*?)</(?:h2|p|a)>', blk, re.DOTALL)
                        for t in texts_in_block:
                            clean = re.sub(r'<[^>]+>', '', t).strip()
                            clean = re.sub(r'&[a-z]+;', ' ', clean)
                            clean = re.sub(r'\s+', ' ', clean)
                            if len(clean) > 20:
                                all_snippets.append(clean)
            except Exception:
                continue
        
        if all_snippets:
            seen = set()
            unique = []
            for s in all_snippets:
                key = s[:60]
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            content = ' '.join(unique)
            if len(content) > 8000:
                content = content[:8000]
            try:
                os.makedirs(os.path.dirname(web_file), exist_ok=True)
                with open(web_file, 'w', encoding='utf-8') as f:
                    f.write(content)
            except Exception:
                pass
            logger.info(f"Fallback搜索完成: {len(content)}字, {len(unique)}条摘要")
            return content
        else:
            logger.warning("Fallback搜索无结果")
            return ''
    except Exception as e:
        logger.warning(f"Fallback搜索失败: {e}")
        return ''


def _keyword_score(news_text: str, pos_keywords: list, neg_keywords: list,
                   default: float = 0.50, range_min: float = 0.35, range_max: float = 0.65) -> float:
    """基于关键词计数的简单评分：正向词+分，负向词-分，每个词±0.03"""
    score = default
    for kw in pos_keywords:
        if kw in news_text:
            score += 0.03
    for kw in neg_keywords:
        if kw in news_text:
            score -= 0.03
    return round(np.clip(score, range_min, range_max), 4)


# ============================================================
# 品种级新闻评分 — 打破 D2/D3/D6 全品种同值
# ============================================================
CATEGORY_SENSITIVITY = {
    '黑色系':   {'geopolitical': 0.6, 'policy': 1.4, 'fiscal': 1.5},
    '有色金属': {'geopolitical': 1.3, 'policy': 1.2, 'fiscal': 1.2},
    '贵金属':   {'geopolitical': 1.6, 'policy': 0.7, 'fiscal': 0.5},
    '能化':     {'geopolitical': 1.5, 'policy': 1.0, 'fiscal': 0.7},
    '农产品':   {'geopolitical': 0.7, 'policy': 1.2, 'fiscal': 0.9},
}

def _per_variety_news_score(news_text: str, name: str, code: str, category: str,
                            pos_kw: list, neg_kw: list, sensitivity_key: str) -> float:
    """品种级新闻评分：三层分级
    
    L1 品种直接提及（name/code 在关键词±150字符内 ≥3次）→ 扩评分范围至[0.25,0.75]
    L2 品种有提及或板块提及 → 全局分 × 板块敏感度 × 1.2
    L3 无提及 → 全局分 × 板块敏感度（保守）
    """
    import re
    
    # 收集品种名和代码在新闻中的所有位置
    variety_positions = []
    for term in [name, code.upper()]:
        for m in re.finditer(re.escape(term), news_text):
            variety_positions.append(m.start())
    
    # 统计关键词±150字符窗口内的品种提及次数
    nearby_hits = 0
    all_keywords = pos_kw + neg_kw
    for kw in all_keywords:
        for m in re.finditer(re.escape(kw), news_text):
            kw_pos = m.start()
            for vp in variety_positions:
                if abs(kw_pos - vp) <= 150:
                    nearby_hits += 1
                    break
    
    if nearby_hits >= 3:
        # L1: 品种被明确讨论 → 扩大评分范围，捕捉强信号
        score = _keyword_score(news_text, pos_kw, neg_kw, range_min=0.25, range_max=0.75)
    elif nearby_hits >= 1 or variety_positions:
        # L2: 品种/板块有提及 → 全局分 + 板块敏感度放大
        sensitivity = CATEGORY_SENSITIVITY.get(category, {}).get(sensitivity_key, 1.0)
        base = _keyword_score(news_text, pos_kw, neg_kw)
        score = 0.5 + (base - 0.5) * sensitivity * 1.2
    else:
        # L3: 无提及 → 板块基线
        sensitivity = CATEGORY_SENSITIVITY.get(category, {}).get(sensitivity_key, 1.0)
        base = _keyword_score(news_text, pos_kw, neg_kw)
        score = 0.5 + (base - 0.5) * sensitivity
    
    return round(np.clip(score, 0.30, 0.70), 4)


def _score_d2_geopolitical(name: str = None, code: str = None, category: str = None) -> Tuple[float, str]:
    """D2 地缘政治: Web Search多平台新闻 + SHMET → 合并文本关键词评分
    
    正向词（合作/稳定）：一带一路、RCEP、合作、对话、稳定、和平、自贸
    负向词（冲突/风险）：制裁、关税、贸易战、冲突、地缘、出口管制、脱钩
    """
    merged_text, source = _get_merged_news()
    if not merged_text:
        return -1.0, 'all_news_unavailable'
    pos = ['一带一路', '合作', '对话', '稳定', '和平', '自贸', 'RCEP', '开放']
    neg = ['制裁', '关税', '贸易战', '冲突', '地缘', '出口管制', '脱钩', '紧张']
    if name and code and category:
        score = _per_variety_news_score(merged_text, name, code, category, pos, neg, 'geopolitical')
        return score, f'{source}+per_variety'
    score = _keyword_score(merged_text, pos, neg)
    return score, source


def _score_d3_policy(name: str = None, code: str = None, category: str = None) -> Tuple[float, str]:
    """D3 产业政策: Web Search多平台新闻 + SHMET → 合并文本关键词评分
    
    正向词（利多工业品）：产能扩张、新基建、补贴、碳中和、新能源、制造强国、设备更新
    负向词（利空工业品）：去产能、限产、环保督察、淘汰落后、供给侧改革、高耗能
    """
    merged_text, source = _get_merged_news()
    if not merged_text:
        return -1.0, 'all_news_unavailable'
    pos = ['产能扩张', '新基建', '补贴', '新能源', '制造强国', '设备更新', '产业升级', '数字化转型']
    neg = ['去产能', '限产', '环保督察', '淘汰落后', '供给侧改革', '高耗能', '产能过剩', '整改']
    if name and code and category:
        score = _per_variety_news_score(merged_text, name, code, category, pos, neg, 'policy')
        return score, f'{source}+per_variety'
    score = _keyword_score(merged_text, pos, neg)
    return score, source


def _score_d4_economy() -> Tuple[float, str]:
    """D4 关键经济指标: AKShare PMI+CPI → -1
    
    评分逻辑: PMI>50扩张(利多,>0.55)，CPI 2-3%健康，过高/过低不利
    """
    try:
        import akshare as ak
        logging.getLogger('akshare').setLevel(logging.WARNING)
        
        df_pmi = ak.macro_china_pmi()
        latest_pmi = df_pmi.iloc[0]
        mfg_pmi = float(latest_pmi['制造业-指数'])
        
        df_cpi = ak.macro_china_cpi()
        latest_cpi = df_cpi.iloc[0]
        cpi_yoy = float(latest_cpi['全国-同比增长'])
        
        # PMI评分
        if mfg_pmi >= 52:
            pmi_score = 0.70
        elif mfg_pmi >= 50:
            pmi_score = 0.60
        elif mfg_pmi >= 48:
            pmi_score = 0.45
        else:
            pmi_score = 0.30
        
        # CPI评分: 2-3%最优
        if 2.0 <= cpi_yoy <= 3.0:
            cpi_score = 0.55
        elif 1.0 <= cpi_yoy < 2.0 or 3.0 < cpi_yoy <= 4.0:
            cpi_score = 0.50
        elif cpi_yoy < 0:
            cpi_score = 0.30  # 通缩风险
        elif cpi_yoy > 5:
            cpi_score = 0.35  # 高通胀
        else:
            cpi_score = 0.45
        
        score = round(0.6 * pmi_score + 0.4 * cpi_score, 4)
        source = f'AKShare:PMI={mfg_pmi},CPI_yoy={cpi_yoy}%'
        return score, source
    except Exception as e:
        logger.warning(f"D4 AKShare失败: {e}")
        return -1.0, 'unavailable'


def _score_d5_sentiment() -> Tuple[float, str]:
    """D5 市场情绪: 全市场代表性品种OI变化率中位数 → -1
    
    选取10个代表性品种（覆盖有色/黑色/能化/农产品），
    计算近两日持仓量变化率中位数，映射为市场情绪分。
    OI增长=多头情绪，OI下降=避险情绪。
    至少5个品种有效才出分。
    """
    SENTIMENT_PROXIES = [
        ('CU', '有色'), ('AL', '有色'),
        ('RB', '黑色'), ('I', '黑色'),
        ('SC', '能化'), ('TA', '能化'), ('MA', '能化'),
        ('M', '农产品'), ('CF', '农产品'), ('SR', '农产品'),
    ]
    
    oi_changes = []
    for code, _ in SENTIMENT_PROXIES:
        try:
            kline = _get_akshare_kline(code)
            if kline and len(kline) >= 2:
                latest_oi = kline[-1].get('open_interest', 0)
                prev_oi = kline[-2].get('open_interest', 0)
                if prev_oi > 0:
                    oi_changes.append((latest_oi - prev_oi) / prev_oi)
        except Exception:
            continue
    
    if len(oi_changes) >= 5:
        median_change = float(np.median(oi_changes))
        score = round(np.clip(0.5 + median_change * 5, 0.25, 0.75), 4)
        return score, f'AKShare:OI_median_change={median_change:.4f}(n={len(oi_changes)})'
    
    return -1.0, 'unavailable'


def _score_d6_fiscal(name: str = None, code: str = None, category: str = None) -> Tuple[float, str]:
    """D6 财政政策: Web Search多平台新闻 + SHMET → 合并文本关键词评分
    
    正向词（积极财政/利多）：减税降费、财政扩张、专项债、赤字率提高、国债增发、转移支付
    负向词（紧缩财政/利空）：财政紧缩、加税、赤字率下降、削减支出、地方债风险
    """
    merged_text, source = _get_merged_news()
    if not merged_text:
        return -1.0, 'all_news_unavailable'
    pos = ['减税降费', '财政扩张', '专项债', '赤字率', '国债增发', '转移支付', '财政加力', '积极财政']
    neg = ['财政紧缩', '加税', '削减支出', '地方债风险', '财政收紧', '债务风险']
    if name and code and category:
        score = _per_variety_news_score(merged_text, name, code, category, pos, neg, 'fiscal')
        return score, f'{source}+per_variety'
    score = _keyword_score(merged_text, pos, neg)
    return score, source


# ============================================================
# D1-D6 宏观批量评分
# ============================================================
def compute_macro_scores_daily() -> Tuple[dict, dict]:
    """从AKShare API获取D1-D6宏观评分（实时计算，不走缓存）
    
    Returns:
        (scores_dict, sources_dict)
        scores_dict: {dim_key: score}
        sources_dict: {dim_key: source_str}
        
    对于-1（不可用）的维度，由调度层补充Web Search后更新。
    """
    dim_funcs = {
        '1_货币政策': _score_d1_monetary,
        '4_关键经济指标': _score_d4_economy,
        '5_市场情绪': _score_d5_sentiment,
        # D2/D3/D6 改为品种级评分，不再全局计算
    }
    
    scores = {}
    sources = {}
    available_count = 0
    
    for dim_key, func in dim_funcs.items():
        score, source = func()
        scores[dim_key] = score
        sources[dim_key] = source
        if score >= 0:
            available_count += 1
    
    logger.info(f"宏观评分: {available_count}/3维可用(D1/D4/D5)")
    return scores, sources


# ============================================================
# 宏观缓存管理
# ============================================================
def get_macro_cache_path(data_dir: str) -> str:
    return os.path.join(data_dir, 'macro_cache.json')


def load_macro_cache(data_dir: str) -> dict:
    cache_path = get_macro_cache_path(data_dir)
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'r', encoding='utf-8') as f:
        cache = json.load(f)
    cached_date = cache.get('cached_date', '')
    if cached_date:
        try:
            cache_dt = datetime.strptime(cached_date, '%Y-%m-%d')
            if (datetime.now() - cache_dt).days > 7:
                cache['expired'] = True
        except:
            cache['expired'] = True
    return cache


def save_macro_cache(data_dir: str, macro_scores: dict, sources: dict = None):
    cache = {
        'cached_date': datetime.now().strftime('%Y-%m-%d'),
        'cached_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'macro_scores': macro_scores,
        'sources': sources or {},
        'expired': False,
    }
    cache_path = get_macro_cache_path(data_dir)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    logger.info(f"宏观缓存已保存: {cache_path}")


def get_macro_scores(data_dir: str) -> Tuple[dict, dict]:
    """获取宏观维度评分(D1-D6)，优先AKShare实时计算，降级到缓存
    
    Returns:
        (scores_dict, meta_dict)
    """
    # 优先AKShare实时计算
    try:
        scores, sources = compute_macro_scores_daily()
    except Exception as e:
        logger.warning(f"AKShare宏观计算失败: {e}")
        scores, sources = {}, {}
    
    # D2/D3/D6已改为品种级评分，不再从全局缓存读取
    cache = load_macro_cache(data_dir)
    
    if scores:
        save_macro_cache(data_dir, scores, sources)
        available = sum(1 for s in scores.values() if s >= 0)
        return scores, {'available': True, 'source': 'AKShare+cache', 'available_dims': available}
    
    return None, {'available': False, 'source': 'none', 'reason': 'AKShare+缓存均不可用'}


# ============================================================
# D9 基差: AKShare现货 → TQSDK spread代理 → -1
# ============================================================
def _get_spot_basis_akshare(code: str) -> Optional[float]:
    """通过AKShare获取基差率
    
    Returns:
        basis_rate: (spot - futures) / futures, 正值=现货升水
    """
    spot_code = SPOT_VARIETY_MAP.get(code)
    if not spot_code:
        return None
    
    try:
        import akshare as ak
        logging.getLogger('akshare').setLevel(logging.WARNING)
        
        # 取最近5天的现货数据
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
        df = ak.futures_spot_price_daily(start_day=start, end_day=end, vars_list=[spot_code])
        
        if df is None or len(df) == 0:
            return None
        
        # 取最新行
        latest = df.iloc[-1]  # 数据正序（旧→新），取最后一行
        near_basis_rate = latest.get('near_basis_rate')
        dom_basis_rate = latest.get('dom_basis_rate')
        
        # 优先主力合约基差率
        rate = dom_basis_rate if dom_basis_rate and dom_basis_rate == dom_basis_rate else near_basis_rate
        if rate is not None and rate == rate:  # NaN check
            return float(rate)
    except Exception as e:
        logger.debug(f"AKShare现货基差失败 {code}: {e}")
    
    return None


# ============================================================
# 共享TQSDK连接（v3.1：批量评分时复用，避免每品种新建WebSocket）
# ============================================================
_shared_tqsdk_provider = None


def _init_shared_tqsdk():
    """初始化共享TQSDK连接（在score_all_varieties开始时调用）"""
    global _shared_tqsdk_provider
    if _shared_tqsdk_provider is not None:
        return  # 已有连接
    try:
        from tqsdk_data import TQDataProvider, _load_tqsdk_credentials
        u, p = _load_tqsdk_credentials()
        if not u or not p:
            return
        _shared_tqsdk_provider = TQDataProvider(username=u, password=p)
        _shared_tqsdk_provider.connect()
        logger.info("共享TQSDK连接已建立（批量评分复用）")
    except Exception as e:
        logger.warning(f"共享TQSDK连接失败: {e}")
        _shared_tqsdk_provider = None


def _close_shared_tqsdk():
    """关闭共享TQSDK连接（在score_all_varieties结束时调用）"""
    global _shared_tqsdk_provider
    if _shared_tqsdk_provider is not None:
        try:
            _shared_tqsdk_provider.close()
        except Exception:
            pass
        _shared_tqsdk_provider = None


def _get_spread_proxy_tqsdk(code: str) -> Optional[float]:
    """通过TQSDK主力合约价差作为基差代理
    
    优先使用共享连接（批量评分时），否则新建连接。
    价差 = 远月 - 近月, 正值=contango(期货升水=现货贴水=利空), 负值=backwardation(利多)
    """
    # v3.1: 优先复用共享连接
    if _shared_tqsdk_provider is not None:
        try:
            spread = _shared_tqsdk_provider.get_main_spread(code)
            if spread and spread.get('near_price') and spread.get('far_price'):
                near = spread['near_price']
                far = spread['far_price']
                if near > 0:
                    return (far - near) / near
        except Exception as e:
            logger.debug(f"共享TQSDK价差失败 {code}: {e}")
        return None
    
    # 降级：独立连接（单品种调用时使用）
    try:
        from tqsdk_data import TQDataProvider, _load_tqsdk_credentials
        u, p = _load_tqsdk_credentials()
        if not u or not p:
            return None
        
        with TQDataProvider(username=u, password=p) as prov:
            spread = prov.get_main_spread(code)
            if spread and spread.get('near_price') and spread.get('far_price'):
                near = spread['near_price']
                far = spread['far_price']
                if near > 0:
                    return (far - near) / near
    except Exception as e:
        logger.debug(f"TQSDK价差代理失败 {code}: {e}")
    return None


def _score_d9_basis(code: str, name: str = None) -> Tuple[float, str]:
    """D9 基差: AKShare现货 → TQSDK spread → Web Search → -1
    
    basis_rate > 0 (现货升水) = 利多 → score > 0.5
    basis_rate < 0 (现货贴水) = 利空 → score < 0.5
    """
    # 第一优先: AKShare现货基差
    basis_rate = _get_spot_basis_akshare(code)
    if basis_rate is not None:
        # 基差率映射: ±5% → score 0.3-0.7
        score = round(np.clip(0.5 + basis_rate * 4, 0.2, 0.8), 4)
        return score, f'AKShare:basis_rate={basis_rate:.4f}'
    
    # 第二优先: TQSDK价差代理
    spread_pct = _get_spread_proxy_tqsdk(code)
    if spread_pct is not None:
        # spread_pct正值=contango=期货升水=利空, 反转符号
        score = round(np.clip(0.5 - spread_pct * 3, 0.25, 0.75), 4)
        return score, f'TQSDK:spread_pct={spread_pct:.4f}'
    
    # 第三层: Web Search兜底（v4.9新增，TQSDK WebSocket永久不可用后的降级）
    if name:
        ws_score = _d9_web_search_fallback(code, name)
        if ws_score is not None:
            return ws_score, f'WebSearch:fallback'
    
    return -1.0, 'unavailable'


# ============================================================
# D9 基差缓存（v4.10新增 — Agent预取，Python脚本只读）
# ============================================================
_D9_CACHE = None
_D9_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'd9_basis_cache.json')


def _load_d9_cache() -> dict:
    """加载D9基差预取缓存（仅首次调用时读文件）"""
    global _D9_CACHE
    if _D9_CACHE is not None:
        return _D9_CACHE
    try:
        cache_path = os.path.normpath(_D9_CACHE_PATH)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as f:
                _D9_CACHE = json.load(f)
            cache_date = _D9_CACHE.get('_meta', {}).get('last_updated', 'unknown')
            logger.info(f"D9缓存已加载: {cache_date}, {len(_D9_CACHE.get('varieties',{}))}品种")
            return _D9_CACHE
    except Exception as e:
        logger.warning(f"D9缓存加载失败: {e}")
    _D9_CACHE = {}
    return _D9_CACHE


# ============================================================
# D9 Web Search降级（降级链第三层，v4.9新增 → v4.10加入缓存）
# ============================================================
def _d9_web_search_fallback(code: str, name: str) -> Optional[float]:
    """D9 Web Search兜底 — 缓存→Bing搜索→定性判断
    
    当AKShare futures_spot_price_daily 和 TQSDK spread 都失败时调用。
    v4.10: 优先读取Agent预取缓存(JSON文件)，缓存miss后再尝试Bing搜索。
    返回[0.2,0.8]的score或None。
    """
    # === v4.10 第一优先：Agent预取缓存 ===
    cache = _load_d9_cache()
    var_data = cache.get('varieties', {}).get(code)
    if var_data and var_data.get('confidence', 0) >= 0.5:
        basis_rate = var_data.get('basis_rate')
        if basis_rate is not None and basis_rate == basis_rate:  # not NaN
            score = round(np.clip(0.5 + basis_rate * 4, 0.2, 0.8), 4)
            logger.info(f"D9缓存命中: {name}({code}) basis_rate={basis_rate:.4f} → score={score:.4f}")
            return score
    
    # === 第二优先：Bing搜索（原逻辑） ===
    try:
        # 构造搜索词: 品种名 + 现货 + 基差/升贴水
        keywords = f'{name} 现货价格 基差 升贴水 2026'
        
        resp = requests.get(
            'https://www.bing.com/search',
            params={'q': keywords},
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        
        text = resp.text
        
        # 策略1: 直接搜索基差率/升贴水百分比
        import re as _re
        # 匹配 "基差率 -2.3%" "升水 150元/吨" "贴水 -80" 等模式
        basis_patterns = [
            _re.findall(r'基差[率]?\s*[：:]\s*([+-]?\d+\.?\d*)\s*%?', text),
            _re.findall(r'升贴水\s*[：:]\s*([+-]?\d+\.?\d*)', text),
            _re.findall(r'现货.{0,10}(升水|贴水).{0,10}([+-]?\d+\.?\d*)', text),
        ]
        
        for matches in basis_patterns:
            for m in matches:
                try:
                    val = float(m) if isinstance(m, str) else float(m[1] if isinstance(m, tuple) else m)
                    # 合理范围: -20%到+20%基差率, 或 -1000到+1000元/吨
                    if -20 <= val <= 20 and val != 0:
                        # 正值=升水=利多, 负值=贴水=利空
                        score = round(np.clip(0.5 + val * 0.04, 0.25, 0.75), 4)
                        logger.info(f"D9 WebSearch: {name} basis≈{val:.2f}% → score={score:.4f}")
                        return score
                except (ValueError, TypeError):
                    continue
        
        # 策略2: 文本中搜索"现货升水"/"现货贴水"定性判断
        if '现货升水' in text or '大幅升水' in text or '严重升水' in text:
            # 有升水描述但未能量化 → 偏多0.55
            logger.info(f"D9 WebSearch: {name} 定性=现货升水 → score=0.55")
            return 0.55
        elif '现货贴水' in text or '大幅贴水' in text or '严重贴水' in text:
            # 有贴水描述但未能量化 → 偏空0.45
            logger.info(f"D9 WebSearch: {name} 定性=现货贴水 → score=0.45")
            return 0.45
        
        return None
    except Exception as e:
        logger.debug(f"D9 Web Search降级失败 {name}({code}): {e}")
        return None


# ============================================================
# D10 跨市场联动: AKShare外盘历史 → 趋势对比+隔夜信号
# ============================================================
def _compute_single_overseas(code: str, symbol: str, closes: np.ndarray) -> Tuple[float, str]:
    """计算单个外盘品种的D10子得分"""
    try:
        if symbol not in _overseas_cache:
            import akshare as ak
            logging.getLogger('akshare').setLevel(logging.WARNING)
            df = ak.futures_foreign_hist(symbol=symbol)
            if df is None or len(df) < 10:
                _overseas_cache[symbol] = None
            else:
                _overseas_cache[symbol] = df['close'].values.astype(float)
        
        overseas_closes = _overseas_cache.get(symbol)
        if overseas_closes is None:
            # v3.2: AKShare失败 → Web Search降级（第三层）
            ws_signal = _d10_web_search_fallback(symbol)
            if ws_signal is not None:
                d10 = float(np.clip(0.5 + ws_signal * 0.3, 0.2, 0.8))
                return d10, f'WebSearch:{symbol}(signal={ws_signal:.3f})'
            return -1.0, 'unavailable'
        
        # --- 信号①: 趋势对比 (60%) ---
        if len(closes) >= 10 and len(overseas_closes) >= 10:
            dom_10d_pct = (closes[-1] - closes[-10]) / (abs(closes[-10]) + 1e-8)
            ovs_10d_pct = (overseas_closes[-1] - overseas_closes[-10]) / (abs(overseas_closes[-10]) + 1e-8)
            trend_diff = ovs_10d_pct - dom_10d_pct
            trend_score = 0.5 + np.tanh(trend_diff * 3) * 0.3
        else:
            trend_score = 0.5
            trend_diff = 0
        
        # --- 信号②: 隔夜信号 (40%) ---
        if len(overseas_closes) >= 2:
            overnight_pct = (overseas_closes[-1] - overseas_closes[-2]) / (abs(overseas_closes[-2]) + 1e-8)
            overnight_score = 0.5 + np.tanh(overnight_pct * 5) * 0.3
        else:
            overnight_score = 0.5
            overnight_pct = 0
        
        d10 = np.clip(0.6 * trend_score + 0.4 * overnight_score, 0.2, 0.8)
        return float(d10), f'{symbol}(td={trend_diff:.4f},on={overnight_pct:.4f})'
    except Exception as e:
        logger.warning(f"D10外盘失败 {code}:{symbol}: {e}")
        return -1.0, 'unavailable'


def _score_d10_cross_market(code: str, closes: np.ndarray) -> Tuple[float, str]:
    """D10 跨市场联动: AKShare外盘历史趋势对比
    
    两个信号加权:
    ① 趋势对比(60%): 同窗口10日涨跌幅对比，外盘领先=利多，外盘落后=利空
    ② 隔夜信号(40%): 外盘最近一日涨跌，休市期间上涨=次日偏多
    
    多市场融合: cu同时拉取LME铜(CAD)+COMEX铜(HG)，等权平均
    无外盘映射=0.5, 全部数据获取失败=-1
    """
    overseas = OVERSEAS_MAP.get(code)
    if overseas is None:
        return 0.5, 'no_overseas_mapping_neutral'
    
    # 统一为列表
    symbols = overseas if isinstance(overseas, list) else [overseas]
    
    scores, sources = [], []
    for sym in symbols:
        s, src = _compute_single_overseas(code, sym, closes)
        if s >= 0:  # 只计入成功的（-1=失败）
            scores.append(s)
            sources.append(src)
    
    if not scores:
        return -1.0, 'overseas_unavailable'
    
    # 多市场等权平均
    d10 = round(np.mean(scores), 4)
    source_str = 'AKShare:' + '|'.join(sources)
    return float(d10), source_str


# ============================================================
# D14 价差结构: AKShare近远月 → TQSDK get_main_spread → -1
# ============================================================
_spread_akshare_cache = {}

def _get_spread_from_akshare(code: str) -> Optional[float]:
    """通过AKShare获取近远月合约价差
    
    获取近月和远月合约的最新收盘价，计算 (far-near)/near
    """
    global _spread_akshare_cache
    if code in _spread_akshare_cache:
        return _spread_akshare_cache[code]
    
    try:
        from tqsdk_symbols import get_main_months
        import akshare as ak
        
        months = get_main_months(code)
        if len(months) < 2:
            _spread_akshare_cache[code] = None
            return None
        
        now = datetime.now()
        current_month = now.month
        year = now.year
        
        months_int = sorted([int(m) for m in months])
        near_m, far_m = None, None
        for m in months_int:
            if m >= current_month and near_m is None:
                near_m = m
            elif near_m is not None:
                far_m = m
                break
        
        if near_m is None:
            near_m = months_int[0]
        if far_m is None:
            far_m = months_int[1] if len(months_int) > 1 else (near_m % 12) + 1
        
        near_code = f'{code.upper()}{year%100:02d}{near_m:02d}'
        far_code = f'{code.upper()}{year%100:02d}{far_m:02d}'
        
        df_near = ak.futures_zh_daily_sina(symbol=near_code)
        time.sleep(0.3)
        df_far = ak.futures_zh_daily_sina(symbol=far_code)
        
        if df_near is None or df_far is None or len(df_near) == 0 or len(df_far) == 0:
            _spread_akshare_cache[code] = None
            return None
        
        near_close = float(df_near.iloc[-1]['close'])
        far_close = float(df_far.iloc[-1]['close'])
        
        if near_close > 0:
            spread_pct = (far_close - near_close) / near_close
            _spread_akshare_cache[code] = spread_pct
            return spread_pct
    except Exception as e:
        logger.debug(f"AKShare价差失败 {code}: {e}")
    
    _spread_akshare_cache[code] = None
    return None


def _score_d14_spread_structure(code: str) -> Tuple[float, str]:
    """D14 价差结构: AKShare近远月 → TQSDK get_main_spread → -1
    
    spread > 0 (contango/正向市场) = 供应充足/需求偏弱 → score < 0.5
    spread < 0 (backwardation/反向市场) = 供应偏紧/需求旺盛 → score > 0.5
    """
    # 第一优先: AKShare近远月合约收盘价差（免费无连接限制）
    spread_pct = _get_spread_from_akshare(code)
    if spread_pct is not None:
        score = round(np.clip(0.5 - spread_pct * 5, 0.2, 0.8), 4)
        return score, f'AKShare:spread_pct={spread_pct:.4f}'
    
    # 第二优先: TQSDK实时价差
    spread_pct = _get_spread_proxy_tqsdk(code)
    if spread_pct is not None:
        score = round(np.clip(0.5 - spread_pct * 5, 0.2, 0.8), 4)
        return score, f'TQSDK:spread_pct={spread_pct:.4f}'
    
    return -1.0, 'unavailable'


# ============================================================
# D7/D8 库存数据获取（AKShare futures_inventory_em）
# ============================================================

# 品种代码→futures_inventory_em 中文名映射（53/55 品种覆盖，原油sc/集运ec不支持）
_INVENTORY_EM_NAME_MAP = {
    'rb': '螺纹钢', 'hc': '热卷', 'i': '铁矿石', 'jm': '焦煤', 'j': '焦炭',
    'sf': '硅铁', 'sm': '锰硅', 'cu': '沪铜', 'al': '沪铝', 'zn': '沪锌',
    'ni': '镍', 'sn': '锡', 'ao': '氧化铝', 'si': '工业硅',
    'au': '沪金', 'ag': '沪银',
    'fu': '燃油', 'lu': '低硫燃料油', 'bu': '沥青', 'ru': '橡胶', 'nr': '20号胶',
    'ta': 'PTA', 'ma': '甲醇', 'sa': '纯碱', 'fg': '玻璃', 'pp': '聚丙烯',
    'l': '塑料', 'v': 'PVC', 'eg': '乙二醇', 'pf': '短纤', 'eb': '苯乙烯',
    'sp': '纸浆', 'ur': '尿素', 'lc': '碳酸锂',
    'a': '豆一', 'b': '豆二', 'm': '豆粕', 'y': '豆油', 'p': '棕榈',
    'oi': '菜油', 'rm': '菜粕', 'c': '玉米', 'cs': '玉米淀粉',
    'lh': '生猪', 'jd': '鸡蛋', 'cf': '郑棉', 'sr': '白糖',
    'ap': '苹果', 'cj': '红枣', 'pk': '花生', 'cy': '棉纱',
    'pb': '沪铅', 'ss': '不锈钢',
}
# sc(原油)和ec(集运指数)不在futures_inventory_em覆盖范围，D7/D8返回-1


def _get_warehouse_inventory(code: str) -> Optional[np.ndarray]:
    """从AKShare获取品种仓单库存时序
    
    Args:
        code: 品种代码（小写，如'al'）

    Returns:
        np.ndarray: 库存时序（最新在最后），或None
    """
    em_name = _INVENTORY_EM_NAME_MAP.get(code)
    if em_name is None:
        # v3.2: sc原油 → EIA库存代理（futures_inventory_em不覆盖原油）
        if code == 'sc':
            eia_inv = _get_eia_crude_inventory()
            if eia_inv is not None and len(eia_inv) >= 4:
                # EIA周度→日度线性插值（5周→25个交易日）
                n_weekly = len(eia_inv)
                n_daily = n_weekly * 5
                x_weekly = np.arange(n_weekly) * 5
                x_daily = np.arange(n_daily)
                daily_inv = np.interp(x_daily, x_weekly, eia_inv)
                return daily_inv
        # ec(集运)结构性无库存指标，返回None（D7/D8=-1）
        return None
    try:
        import akshare as ak
        df = ak.futures_inventory_em(symbol=em_name)
        if df is None or len(df) == 0:
            return None
        inventory = df['库存'].values.astype(float)
        return inventory
    except Exception as e:
        logger.debug(f"futures_inventory_em({em_name})失败: {e}")
        return None


def _score_d7_supply_inventory(code: str) -> Tuple[float, str]:
    """D7 供应端/库存: AKShare仓单库存 → -1
    
    评分逻辑：
    - 库存近5日下降 → 供应偏紧 → 利多（>0.5）
    - 库存近5日上升 → 供应充裕 → 利空（<0.5）
    - 库存处于20日低位 → 偏多；高位 → 偏空
    """
    inv = _get_warehouse_inventory(code)
    if inv is None or len(inv) < 6:
        return -1.0, 'inventory_unavailable'

    n = len(inv)
    latest = inv[-1]
    
    # 5日库存变化率
    inv_5d_ago = inv[-6]
    if inv_5d_ago > 0:
        inv_change_5d = (latest - inv_5d_ago) / inv_5d_ago
    else:
        inv_change_5d = 0.0
    
    # 20日库存分位（低分位=库存低=利多）
    if n >= 20:
        inv_20d_min = np.min(inv[-20:])
        inv_20d_max = np.max(inv[-20:])
        if inv_20d_max > inv_20d_min:
            inv_percentile = (latest - inv_20d_min) / (inv_20d_max - inv_20d_min)
        else:
            inv_percentile = 0.5
    else:
        inv_percentile = 0.5
    
    # 变化率→评分（下降利多，上升利空）
    change_score = np.clip(0.5 - inv_change_5d * 5, 0.2, 0.8)
    # 分位→评分（低分位利多，高分位利空）
    pct_score = np.clip(0.5 - (inv_percentile - 0.5) * 0.6, 0.25, 0.75)
    
    d7 = round(0.55 * change_score + 0.45 * pct_score, 4)
    d7 = float(np.clip(d7, 0.15, 0.85))
    return d7, f'AKShare:warehouse_inv(Δ5d={inv_change_5d:.3f},pct={inv_percentile:.2f})'


def _score_d8_demand_balance(code: str, closes: np.ndarray) -> Tuple[float, str]:
    """D8 需求端/供需平衡: AKShare仓单库存 × K线价格联动(20日窗口) → -1
    
    v3.2 (2026-06-09): 窗口从5日→20日，与D7(5日)形成短/中周期互补
      D7 回答"本周库存压力"，D8 回答"本月供需格局"
      解决D7↔D8共线性(r=0.8925)，库存信号不再被重复放大
    
    评分逻辑（库存+价格联合判断供需，20日窗口）：
    - 库存↓+价格↑ = 需求旺盛（0.60-0.80）
    - 库存↓+价格↓ = 需求疲软（0.35-0.50）
    - 库存↑+价格↑ = 供给压力但需求承接（0.45-0.60）
    - 库存↑+价格↓ = 供过于求（0.20-0.40）
    """
    inv = _get_warehouse_inventory(code)
    window = 20
    if inv is None or len(inv) < window + 1 or len(closes) < window + 1:
        return -1.0, 'inventory_or_kline_unavailable'

    latest_inv = inv[-1]
    inv_20d_ago = inv[-(window + 1)]
    inv_change = (latest_inv - inv_20d_ago) / (inv_20d_ago + 1e-8)
    
    price_change_20d = (closes[-1] - closes[-(window + 1)]) / (abs(closes[-(window + 1)]) + 1e-8)
    
    # 供需四象限（20日窗口，阈值保持±1%库存 ±0.5%价格）
    inv_down = inv_change < -0.01   # 库存下降>1%
    inv_up = inv_change > 0.01      # 库存上升>1%
    price_up = price_change_20d > 0.005
    price_down = price_change_20d < -0.005
    
    if inv_down and price_up:
        base = 0.70  # 需求旺盛
    elif inv_down and price_down:
        base = 0.42  # 需求疲软
    elif inv_up and price_up:
        base = 0.52  # 供给压力但需求承接
    elif inv_up and price_down:
        base = 0.30  # 供过于求
    else:
        base = 0.50  # 变化不大，中性
    
    # 用库存变化幅度和价格变化幅度微调（20日窗口幅度更大，降低系数防过调）
    inv_adj = np.clip(-inv_change * 0.8, -0.08, 0.08)
    price_adj = np.clip(price_change_20d * 0.6, -0.08, 0.08)
    
    d8 = round(np.clip(base + inv_adj + price_adj, 0.15, 0.85), 4)
    return float(d8), f'AKShare:warehouse_inv+price_20d(Δinv={inv_change:.3f},Δprice={price_change_20d:.3f})'


# ============================================================
# D7-D14 产业层计算
# ============================================================
def compute_industrial_scores(
    code: str,
    kline: List[dict],
    month: int,
    category: str,
    precomputed_tech: List[float] = None,
) -> Dict[int, Tuple[float, str]]:
    """从K线数据+外部数据源计算D7-D14产业维度评分
    
    每个维度返回 (score, source)
    score=-1表示不可用
    
    D7 供应端: AKShare仓单库存(futures_inventory_em) → -1
    D8 需求端: AKShare仓单库存×价格联动(四象限) → -1
    D9 基差: AKShare现货 → TQSDK spread → -1
    D10 跨市场: AKShare外盘趋势对比+隔夜信号
    D11 价格位置: 技术面综合(复用预计算)
    D12 周期性: 品种级历史回测季节性
    D13 资金面: K线OI持仓流 → 成交量代理 → -1
    D14 价差结构: AKShare近远月 → TQSDK → -1
    """
    if not kline or len(kline) < 20:
        return None
    
    closes = np.array([k['close'] for k in kline], dtype=float)
    opens = np.array([k['open'] for k in kline], dtype=float)
    highs = np.array([k['high'] for k in kline], dtype=float)
    lows = np.array([k['low'] for k in kline], dtype=float)
    volumes = np.array([k['volume'] for k in kline], dtype=float)
    ois = np.array([k.get('open_interest') or 0 for k in kline], dtype=float)
    
    n = len(closes)
    latest_c = closes[-1]
    ma20 = np.mean(closes[-20:])
    
    result = {}
    
    # ---- D7 供应端: AKShare仓单库存（v3.1 修复：移除K线伪推断，用真实库存数据） ----
    d7_score, d7_source = _score_d7_supply_inventory(code)
    result[7] = (d7_score, d7_source)
    
    # ---- D8 需求端: AKShare仓单库存×价格联动（v3.1 修复：移除K线伪推断，用库存+价格四象限） ----
    d8_score, d8_source = _score_d8_demand_balance(code, closes)
    result[8] = (d8_score, d8_source)
    
    # ---- D9 基差: AKShare现货 → TQSDK spread → Web Search → -1 ----
    d9_score, d9_source = _score_d9_basis(code, name)
    result[9] = (d9_score, d9_source)
    
    # ---- D10 跨市场: AKShare外盘趋势对比+隔夜信号 ----
    d10_score, d10_source = _score_d10_cross_market(code, closes)
    result[10] = (d10_score, d10_source)
    
    # ---- D11 价格位置: 技术面综合（P2修复：复用预计算技术面，避免重复调用） ----
    # v4.6修复: 检查precomputed_tech有效性（-1表示技术面不可用，不要复用）
    if precomputed_tech and len(precomputed_tech) >= 12 and precomputed_tech[0] >= 0:
        d11 = 0.3 * precomputed_tech[0] + 0.2 * precomputed_tech[1] + 0.25 * precomputed_tech[10] + 0.25 * precomputed_tech[11]
        d11 = round(np.clip(d11, 0, 1), 4)
        result[11] = (d11, 'Kline:tech_composite(cached)')
    else:
        try:
            tech = compute_technical_scores(
                opens.tolist(), highs.tolist(), lows.tolist(),
                closes.tolist(), volumes.tolist(), ois.tolist()
            )
            if tech and len(tech) >= 12:
                d11 = 0.3 * tech[0] + 0.2 * tech[1] + 0.25 * tech[10] + 0.25 * tech[11]
                d11 = round(np.clip(d11, 0, 1), 4)
                result[11] = (d11, 'Kline:tech_composite')
            else:
                result[11] = (-1.0, 'tech_failed')
        except Exception as e:
            logger.debug(f"D11技术面计算失败: {e}")
            result[11] = (-1.0, 'tech_failed')
    
    # ---- D12 周期性: 历史回测季节性 ----
    _load_seasonality_cache()
    d12 = _compute_seasonality(code, month)
    _save_seasonality_cache()
    result[12] = (round(d12, 4), f'seasonality:m{month}_{code.lower()}')
    
    # ---- D13 资金面: 持仓量变化 → 成交量代理 → -1 ----
    # v4.2 修复: 期货语境下OI变化不能独立加总，应作为价格方向的放大器
    #   OI↑+Price↑=多头加仓(偏多)   OI↑+Price↓=空头加码(偏空)
    #   OI↓+Price↑=空头离场(偏多)   OI↓+Price↓=多头离场(偏空)
    # TQSDK主力合约K线不返回open_interest，此时用成交量变化作为代理
    has_valid_oi = n >= 10 and ois[-10] > 0
    if has_valid_oi:
        oi_change_10d = (ois[-1] - ois[-10]) / ois[-10]
        oi_volatility = np.std(ois[-10:]) / (np.mean(ois[-10:]) + 1e-8)
        price_change_10d = (closes[-1] - closes[-10]) / (abs(closes[-10]) + 1e-8)
        # v4.2: OI变化放大价格方向——增仓强化信号方向，减仓弱化
        oi_amplifier = 1.0 + 3.0 * abs(oi_change_10d)
        flow_score = 0.5 + 0.5 * np.tanh(price_change_10d * 10 * oi_amplifier)
        d13 = 0.7 * flow_score + 0.3 * (1 - min(oi_volatility * 5, 1))
        result[13] = (round(np.clip(d13, 0, 1), 4), 'Kline:OI_flow')
    elif n >= 10 and np.sum(volumes[-10:]) > 0:
        # 用成交量变化作为资金面代理（成交量方向语义弱，仅做轻度修正）
        vol_change_10d = (np.mean(volumes[-5:]) - np.mean(volumes[-10:-5])) / (np.mean(volumes[-10:-5]) + 1e-8)
        price_change_10d = (closes[-1] - closes[-10]) / (abs(closes[-10]) + 1e-8)
        vol_amplifier = 1.0 + 1.5 * abs(vol_change_10d)
        flow_score = 0.5 + 0.5 * np.tanh(price_change_10d * 8 * vol_amplifier)
        d13 = round(np.clip(flow_score, 0.2, 0.8), 4)
        result[13] = (d13, 'Kline:vol_proxy')
    else:
        result[13] = (-1.0, 'insufficient_data')
    
    # ---- D14 价差结构: AKShare → TQSDK → -1 ----
    d14_score, d14_source = _score_d14_spread_structure(code)
    result[14] = (d14_score, d14_source)
    
    return result


# ============================================================
# 技术面评分（CA内部调用）
# ============================================================
def _compute_tech_scores(kline: List[dict]) -> List[float]:
    """从K线计算24维技术面评分，归入CA统一输出

    Args:
        kline: K线列表（≥20条）

    Returns:
        list[float]: 24维评分 [0,1] 或 -1.0（不可用）。K线不足/计算异常均返回 -1.0，严禁返回中性值 0.5
    """
    if not kline or len(kline) < 20:
        return [-1.0] * 24

    try:
        opens = [k['open'] for k in kline]
        highs = [k['high'] for k in kline]
        lows = [k['low'] for k in kline]
        closes = [k['close'] for k in kline]
        volumes = [k.get('volume', 0) or 0 for k in kline]
        ois = [k.get('open_interest', 0) or 0 for k in kline]

        return compute_technical_scores(opens, highs, lows, closes, volumes, ois)
    except Exception as e:
        logger.warning(f"技术面计算失败: {e}")
        return [-1.0] * 24


# ============================================================
# 主入口：计算单个品种完整38维CA评分
# ============================================================
def score_single_variety(
    vid: int,
    kline: List[dict],
    month: int,
    macro_scores: dict,
) -> Tuple[dict, dict]:
    """计算单个品种的完整38维CA评分（14基本面 + 24技术面）

    Args:
        vid: 品种ID (0-52)
        kline: K线列表（50条日K）
        month: 当前月份
        macro_scores: 宏观缓存评分 {dim_key: score}

    Returns:
        (scores_dict, meta_dict)
        scores_dict: {1: score, ..., 38: score}  1-14基本面, 15-38技术面
        meta_dict: {'source_d1': str, ...}
    """
    name, code, category = get_variety_info(vid)
    meta = {}
    scores = {}

    # --- D1/D4/D5: 全局宏观层（所有品种共享） ---
    for dim_id, dim_name in [(1, '货币政策'), (4, '关键经济指标'), (5, '市场情绪')]:
        key = f'{dim_id}_{dim_name}'
        scores[dim_id] = macro_scores.get(key, -1.0) if macro_scores else -1.0
        meta[f'd{dim_id}_source'] = 'macro_cache' if macro_scores and macro_scores.get(key, -1) >= 0 else 'unavailable'
    
    # --- D2/D3/D6: 品种级新闻评分（打破全品种同值） ---
    d2_score, d2_src = _score_d2_geopolitical(name, code, category)
    d3_score, d3_src = _score_d3_policy(name, code, category)
    d6_score, d6_src = _score_d6_fiscal(name, code, category)
    scores[2] = d2_score
    scores[3] = d3_score
    scores[6] = d6_score
    meta['d2_source'] = d2_src
    meta['d3_source'] = d3_src
    meta['d6_source'] = d6_src

    # --- T1-T24: 技术面（v3.0：归入CA统一输出） ---
    tech_scores = _compute_tech_scores(kline)

    # --- D7-D14: 产业层（P2修复：传入预计算技术面，避免D11重复调用compute_technical_scores） ---
    industrial = compute_industrial_scores(code, kline, month, category, precomputed_tech=tech_scores)
    if industrial:
        for dim_id, (score, source) in industrial.items():
            scores[dim_id] = score
            meta[f'd{dim_id}_source'] = source
    else:
        for i in range(7, 15):
            scores[i] = -1.0
            meta[f'd{i}_source'] = 'unavailable'

    # --- T1-T24: 技术面 → scores[15-38]（已在上面预计算，复用） ---
    tech_available = all(ts >= 0 for ts in tech_scores)
    for i, ts in enumerate(tech_scores):
        scores[15 + i] = ts  # dim15=趋势方向, ..., dim38=向下突破强度
        meta[f'd{15+i}_source'] = 'technical_indicators' if tech_available else 'unavailable'

    meta['scored_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return scores, meta


# ============================================================
# 批量评分：全品种14维CA评分
# ============================================================
def score_all_varieties(
    data_dir: str,
    month: int = None,
    verbose: bool = True,
) -> Tuple[Dict[int, dict], dict]:
    """全品种批量CA评分
    
    Returns:
        (all_scores, summary)
    """
    if month is None:
        month = datetime.now().month
    
    # P0修复: 日期变更时自动重置全部模块级缓存
    _ensure_cache_fresh()
    
    # v3.1: 初始化共享TQSDK连接（批量评分复用，避免每品种新建WebSocket超时）
    _init_shared_tqsdk()
    
    try:
        # Step 1: 宏观评分（AKShare实时 → 缓存降级）
        macro_scores, macro_meta = get_macro_scores(data_dir)
    
        if verbose:
            if macro_scores:
                available = sum(1 for s in macro_scores.values() if s >= 0)
                print(f"[CA Scorer] 宏观: {available}/3维可用(D1/D4/D5), 来源={macro_meta.get('source','?')}")
            else:
                print(f"[CA Scorer] ⚠️ 宏观层完全不可用")
    
        # Step 2: TQSDK批量获取K线
        from tqsdk_data import get_klines_for_technical
    
        all_scores = {}
        all_meta = {}
        tqsdk_ok = 0
        akshare_ok = 0
        all_fail = 0
        unavailable = 0
        dim_stats = {i: {'available': 0, 'total': NUM_VARIETIES} for i in range(1, 39)}  # 38维
    
        for vid in range(NUM_VARIETIES):
            name, code, category = VARIETY_INFO[vid]
        
            kline = None
            kline_source = 'unavailable'
        
            # AKShare → 降级链第一层（免费无连接限制）
            kline = _get_akshare_kline(code)
            if kline and len(kline) >= 20:
                kline_source = 'AKShare'
                akshare_ok += 1
            else:
                kline = None
        
            # TQSDK降级
            if kline is None:
                try:
                    kline = get_klines_for_technical(code)
                    if kline and len(kline) >= 20:
                        kline_source = 'TQSDK'
                        tqsdk_ok += 1
                        if verbose and akshare_ok + all_fail < NUM_VARIETIES:
                            print(f"  [{name}] AKShare失败→TQSDK降级")
                    else:
                        kline = None
                except:
                    pass
        
            # Sina nf_ 降级（v3.2新增：第三层，免费HTTP接口）
            if kline is None:
                kline = _get_sina_kline(code)
                if kline and len(kline) >= 20:
                    kline_source = 'Sina'
                    if verbose:
                        print(f"  [{name}] AKShare+TQSDK失败→Sina降级")
                else:
                    kline = None
        
            if kline is None:
                all_fail += 1
        
            if kline and len(kline) >= 20:
                scores, meta = score_single_variety(vid, kline, month, macro_scores)
            else:
                unavailable += 1
                scores = {i: -1.0 for i in range(1, 39)}
                meta = {'source': 'unavailable', 'reason': 'AKShare+TQSDK均失败'}
        
            # 统计各维度可用性（38维）
            for dim_id in range(1, 39):
                if scores.get(dim_id, -1) >= 0:
                    dim_stats[dim_id]['available'] += 1
        
            all_scores[vid] = scores
            all_meta[vid] = meta
    
        # 统计
        dim_available = {i: dim_stats[i]['available'] for i in range(1, 39)}
    
        summary = {
            'total': NUM_VARIETIES,
            'tqsdk_ok': tqsdk_ok,
            'akshare_ok': akshare_ok,
            'unavailable': unavailable,
            'macro_available': macro_scores is not None,
            'macro_meta': macro_meta,
            'dim_availability': dim_available,
            'scored_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    
        if verbose:
            print(f"[CA Scorer] K线: AKShare={akshare_ok}, TQSDK={tqsdk_ok}, 失败={unavailable}")
            # 显示各维度覆盖率
            weak_dims = [i for i, a in dim_available.items() if a < 10]
            if weak_dims:
                dim_names = ['货币政策','地缘政治','产业政策','关键经济指标','市场情绪','财政政策',
                            '供应','需求','基差','跨市场','价格位置','周期性','资金面','价差结构']
                print(f"[CA Scorer] ⚠️ 覆盖率偏低: " + 
                      ", ".join(f"D{i}({dim_names[i-1]})={dim_available[i]}/{NUM_VARIETIES}" for i in weak_dims))
    
        return all_scores, summary
    
    finally:
        _close_shared_tqsdk()


# ============================================================
# AKShare K线降级
# ============================================================
_ak_cache = {}

def _get_akshare_kline(code: str) -> Optional[List[dict]]:
    global _ak_cache
    if code in _ak_cache:
        return _ak_cache[code]
    
    try:
        import akshare as ak
        import pandas as pd
        
        ak_code = code.upper() + '0'
        df = ak.futures_zh_daily_sina(symbol=ak_code)
        if df is None or len(df) < 20:
            _ak_cache[code] = None
            return None
        
        if 'datetime' in df.columns:
            df['date_col'] = pd.to_datetime(df['datetime'])
        elif 'date' in df.columns:
            df['date_col'] = pd.to_datetime(df['date'])
        else:
            _ak_cache[code] = None
            return None
        
        df = df.sort_values('date_col', ascending=True)
        tail = df.tail(50)
        
        klines = []
        for _, row in tail.iterrows():
            klines.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'open_interest': float(row.get('hold', 0)),
            })
        
        _ak_cache[code] = klines
        return klines
    except Exception as e:
        logger.debug(f"AKShare K线失败 {code}: {e}")
        _ak_cache[code] = None
        return None


# ============================================================
# Sina nf_ K线降级（降级链第三层，v3.2新增）
# ============================================================
_sina_kline_cache = {}

def _get_sina_kline(code: str) -> Optional[List[dict]]:
    """Sina财经期货日线数据 → 降级链第三层
    
    Sina接口免费无需API Key，作为AKShare+TQSDK都失败时的兜底。
    注意: Sina不返回open_interest，OI字段补0。
    
    Args:
        code: 品种代码（小写，如'cu'）

    Returns:
        K线列表 [{open, high, low, close, volume, open_interest}, ...]，失败返回None
    """
    global _sina_kline_cache
    if code in _sina_kline_cache:
        return _sina_kline_cache[code]
    
    try:
        import re as _re
        sina_symbol = f"{code.upper()}0"  # 新浪主力合约代码: CU0, RB0, ...
        url = (
            f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_{sina_symbol}="
            f"/InnerFuturesNewService.getDailyKLine?symbol={sina_symbol}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200 or not resp.text.strip():
            _sina_kline_cache[code] = None
            return None
        
        # JSONP格式: var _CU0=[[...],...]; 提取JSON数组
        text = resp.text.strip()
        # 去掉 var _XXX= 前缀和末尾分号
        json_match = _re.search(r'=\s*(\[.*\])\s*;?\s*$', text, _re.DOTALL)
        if not json_match:
            _sina_kline_cache[code] = None
            return None
        
        raw = json.loads(json_match.group(1))
        if not raw or len(raw) < 20:
            _sina_kline_cache[code] = None
            return None
        
        # Sina格式: [日期, 开盘, 最高, 最低, 收盘, 成交量]
        tail = raw[-50:]  # 最近50条
        klines = []
        for row in tail:
            klines.append({
                'open': float(row[1]),
                'high': float(row[2]),
                'low': float(row[3]),
                'close': float(row[4]),
                'volume': float(row[5]),
                'open_interest': 0.0,  # Sina不返回OI
            })
        
        _sina_kline_cache[code] = klines
        logger.info(f"Sina K线获取成功: {code} ({len(klines)}条)")
        return klines
    except Exception as e:
        logger.debug(f"Sina K线失败 {code}: {e}")
        _sina_kline_cache[code] = None
        return None


# ============================================================
# D10 Web Search降级（降级链第三层，v3.2新增）
# ============================================================
def _d10_web_search_fallback(symbol: str) -> Optional[float]:
    """D10 Web Search兜底 — 搜索外盘期货近期涨跌幅
    
    当AKShare futures_foreign_hist失败时调用。
    从财经网站搜索结果中解析近期涨跌幅，返回[-1,1]方向信号。
    """
    try:
        # 外盘symbol到搜索关键词
        SYMBOL_KEYWORDS = {
            'CAD': 'LME铜 期货 最新行情 涨跌幅',
            'AHD': 'LME铝 期货 最新行情',
            'ZSD': 'LME锌 期货 最新行情',
            'NID': 'LME镍 期货 最新行情',
            'SND': 'LME锡 期货 最新行情',
            'PBD': 'LME铅 期货 最新行情',
            'HG': 'COMEX铜 期货 最新行情',
            'GC': 'COMEX黄金 期货 最新行情',
            'SI': 'COMEX白银 期货 最新行情',
            'CL': 'WTI原油 期货 最新行情',
            'NG': '天然气 期货 最新行情',
        }
        keywords = SYMBOL_KEYWORDS.get(symbol)
        if not keywords:
            return None
        
        resp = requests.get(
            'https://www.bing.com/search',
            params={'q': keywords},
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        
        # 简单解析: 查找百分比数字（如 +1.23% / -0.56%）
        import re as _re
        text = resp.text
        pct_matches = _re.findall(r'([+-]?\d+\.?\d*)\s*%', text)
        if not pct_matches:
            return None
        
        # 取第一个在合理范围内的百分比（-10%~+10%）
        for m in pct_matches:
            try:
                pct = float(m)
                if -10 <= pct <= 10:
                    # 涨跌幅→方向信号（tanh压缩到[-0.3, 0.3]，叠加到0.5基准）
                    return float(np.tanh(pct * 0.5))
            except ValueError:
                continue
        
        return None
    except Exception as e:
        logger.debug(f"D10 Web Search降级失败 {symbol}: {e}")
        return None


# ============================================================
# sc原油 EIA库存代理（v3.2新增）
# ============================================================
def _get_eia_crude_inventory() -> Optional[np.ndarray]:
    """获取EIA原油库存数据（sc/原油专属，EIA官方数据源）
    
    从EIA官方页面解析美国商业原油库存（不含SPR），返回周度时序。
    返回格式兼容 _get_warehouse_inventory: np.ndarray库存序列（千桶，最新在最后）。
    注意: EIA每周三公布，频率低于日度。TTL=24h。
    """
    EIA_URL = 'https://www.eia.gov/dnav/pet/pet_stoc_wstk_dcu_nus_w.htm'
    
    try:
        import pandas as _pd
        tables = _pd.read_html(EIA_URL)
        if not tables or len(tables) < 5:
            return None
        
        # Table[4] 是主数据表，找 "Commercial Crude Oil (Excl. Lease Stock)" 行
        t4 = tables[4]
        t4.columns = ['Product', 'Area'] + list(t4.columns[2:])
        
        for _, row in t4.iterrows():
            prod = str(row['Product'])
            if 'Commercial Crude Oil' in prod and 'Excl. Lease' in prod:
                # 提取周度数据列（跳过最后两列History链接）
                vals = []
                for col in t4.columns[2:-2]:
                    try:
                        v = float(str(row[col]).replace(',', ''))
                        vals.append(v)
                    except (ValueError, TypeError):
                        continue
                if len(vals) >= 4:
                    # 过滤NaN值（最后列常是History链接）
                    clean = [v for v in vals if not np.isnan(v)]
                    if len(clean) >= 4:
                        return np.array(clean, dtype=float)
        return None
        
    except Exception as e:
        logger.debug(f"EIA库存获取失败: {e}")
        return None


# ============================================================
# 将CA评分写入daily_scores CSV
# ============================================================
def write_all_ca_scores(
    all_scores: Dict[int, dict],
    date_str: str,
    scores_dir: str,
) -> int:
    """将38维CA评分写入daily_scores CSV（dim1-dim38）"""
    import csv
    
    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    if not os.path.exists(csv_path):
        logger.error(f"CSV不存在: {csv_path}")
        return 0
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            vid = int(row['variety_id'])
            if vid in all_scores:
                ca = all_scores[vid]
                for i in range(1, 39):
                    row[f'dim{i}'] = f'{ca[i]:.4f}'
            rows.append(row)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    filled = sum(1 for r in rows if float(r.get('dim1', '-1')) >= 0)
    logger.info(f"CA评分已写入: {filled}/{len(rows)}品种")
    return filled


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 测试全局宏观评分（D1/D4/D5） + 品种级新闻评分（D2/D3/D6）
    print("="*50)
    print("D1/D4/D5 全局宏观评分测试")
    scores, sources = compute_macro_scores_daily()
    for i, dim_name in [(1, '货币政策'), (4, '关键经济指标'), (5, '市场情绪')]:
        key = f'{i}_{dim_name}'
        s = scores.get(key, -1)
        src = sources.get(key, '?')
        print(f"  D{i} {dim_name}: {s:.4f} ({src})")
    
    print("\nD2/D3/D6 品种级新闻评分测试 (铜)")
    for dim_id, func in [(2, _score_d2_geopolitical), (3, _score_d3_policy), (6, _score_d6_fiscal)]:
        s, src = func(name='铜', code='cu', category='有色金属')
        print(f"  D{dim_id}: {s:.4f} ({src})")
    
    # 测试D9基差
    print("\n" + "="*50)
    print("D9 基差测试 (CU, RB, SC, M)")
    for code in ['cu', 'rb', 'sc', 'm']:
        score, source = _score_d9_basis(code)
        print(f"  {code}: {score:.4f} ({source})")
    
    # 测试D14价差
    print("\n" + "="*50)
    print("D14 价差结构测试")
    for code in ['cu', 'rb', 'sc', 'm']:
        score, source = _score_d14_spread_structure(code)
        print(f"  {code}: {score:.4f} ({source})")
