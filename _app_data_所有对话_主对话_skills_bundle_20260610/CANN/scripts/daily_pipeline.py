#!/usr/bin/env python3
"""
CANN每日管线脚本 v4.1
每日收盘后执行：预取Web新闻 → 回填昨日y值 → 微调 → CA全维度评分(含技术面) → 追加今日样本 → 推理

核心原则：
- CA统一输出38维评分（14基本面+24技术面），CANN直接消费
- 技术面归入CA引擎内部计算，不再独立采集步骤
- y值来自AKShare次日真实涨跌
- 只有同时具备真实CA+真实y的样本才是有效训练数据
- 今日新采集的CA数据(y=0.5占位)不参与当日微调，等次日回填真实y后再参与

流程（v4.1 简化为5步）：
0. 预取Web新闻：Bing搜索中国商品期货新闻 → CANN/data/web_news.txt（TTL=1小时，失败降级为仅SHMET）
1. 回填昨日y值：用AKShare次日收盘价计算真实涨跌，更新historical_samples.csv
2. 微调模型：仅用已标记样本（真实CA+真实y），今日新样本不参与
3. CA全维度评分（ca_scorer v3.0引擎，含38维）→ 写入daily_scores CSV
4. 追加今日样本 + 全品种推理

调度：每个交易日16:00执行
"""
import os, sys, subprocess

# monkey-patch: 必须在任何依赖导入前执行，修复 pkg_resources 与 Python 3.13+ 的兼容问题
import pkgutil
class DummyImpImporter:
    def find_module(self, fullname, path=None): return None
pkgutil.ImpImporter = DummyImpImporter

# ===== 环境自检：确保依赖包可用 =====
REQUIRED_PKGS = ['numpy', 'pandas', 'akshare', 'tqsdk', 'requests']
for pkg in REQUIRED_PKGS:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[自检] 安装缺失依赖: {pkg}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
        print(f"[自检] {pkg} 安装完成")

import csv, json, time, warnings, logging, re
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import akshare as ak
logging.getLogger('akshare').setLevel(logging.WARNING)

# 项目路径（tqsdk_data.py内部用 from skills.common.xxx 绝对import）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# skills/CANN/scripts → skills/common
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'common'))
# skills/CANN/scripts → 主对话
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
for p in [PROJECT_ROOT, COMMON_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from technical_indicators import compute_technical_scores
from tqsdk_data import get_klines_for_technical
from pretrain_numpy import (NumpyCANNModel, load_pretrained_model, 
                             predict_single, run_daily_training_numpy,
                             load_csv_samples)
from ca_scorer import score_all_varieties, write_all_ca_scores, get_macro_scores

# 品种映射统一从 variety_list 导入
from skills.common.variety_list import get_variety_info, get_variety_category_by_id, NUM_VARIETIES


# ============================================================
# AKShare K线缓存（每日管线只用一次）
# ============================================================
_kline_cache = None

def get_kline_cache() -> Dict[int, pd.DataFrame]:
    """懒加载全品种K线缓存（AKShare → TQSDK降级）"""
    global _kline_cache
    if _kline_cache is not None:
        return _kline_cache

    _kline_cache = {}
    failed = []
    akshare_ok = 0
    tqsdk_ok = 0

    for vid in range(NUM_VARIETIES):
        name, code, _cat = get_variety_info(vid)
        ak_code = code.upper() + '0'
        source = 'unavailable'

        # 第一层：AKShare
        try:
            df = ak.futures_zh_daily_sina(symbol=ak_code)
            if df is not None and len(df) >= 50:
                if 'datetime' in df.columns:
                    df['date_col'] = pd.to_datetime(df['datetime'])
                elif 'date' in df.columns:
                    df['date_col'] = pd.to_datetime(df['date'])
                else:
                    df = None
            else:
                df = None
            if df is not None:
                source = 'AKShare'
                akshare_ok += 1
        except Exception:
            df = None

        # 第二层：TQSDK降级
        if df is None:
            try:
                klines = get_klines_for_technical(code, count=50)
                if klines and len(klines) >= 50:
                    rows = []
                    for k in klines:
                        dt = k.get('datetime')
                        rows.append({
                            'open': k['open'], 'high': k['high'],
                            'low': k['low'], 'close': k['close'],
                            'volume': k['volume'],
                            'hold': k.get('open_interest', 0),
                            'date_col': pd.to_datetime(dt) if dt else None,
                        })
                    df = pd.DataFrame(rows).dropna(subset=['date_col'])
                    if len(df) >= 50:
                        source = 'TQSDK'
                        tqsdk_ok += 1
                    else:
                        df = None
            except Exception:
                df = None

        if df is not None:
            df = df.sort_values('date_col', ascending=True).reset_index(drop=True)
            _kline_cache[vid] = df
        else:
            failed.append(f'{name}({code})')

        time.sleep(0.3)

    summary_parts = [f'AKShare={akshare_ok}/{NUM_VARIETIES}']
    if tqsdk_ok > 0:
        summary_parts.append(f'TQSDK={tqsdk_ok}')
    if failed:
        summary_parts.append(f'失败={len(failed)}')
        summary_parts.append(f'({", ".join(failed[:5])}')
        if len(failed) > 5:
            summary_parts.append(f'...及其他{len(failed)-5}个')
        summary_parts.append(')')
    print(f'  [K线缓存] {", ".join(summary_parts)}')
    return _kline_cache


# ============================================================
# Step 3: 创建评分CSV模板（38维CA评分待填充）
# ============================================================
def generate_today_scores(date_str: str, scores_dir: str) -> int:
    """创建今日评分CSV模板，dim1-dim38初始-1，等待CA评分引擎填充

    v4.0: 技术面已归入CA引擎，此处只创建模板不再独立采集
    
    P2修复：若CSV已存在且有真实CA评分(dim1>=0)，跳过不覆盖。
    """
    print(f'\n[Step 3a] 创建 {date_str} 评分模板')
    
    target_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    month = int(date_str[4:6])
    
    fieldnames = ['date','variety_id','variety_name','month'] + \
                 [f'dim{i}' for i in range(1, 39)]
    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    
    # P2修复：已有真实CA数据时跳过覆盖
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row.get('dim1', '-1')) >= 0:
                    print(f'  ⚠️ {date_str} CSV已存在真实CA评分，跳过覆盖')
                    return NUM_VARIETIES
                break  # 只检查第一行
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vid in range(NUM_VARIETIES):
            name, code, _cat = get_variety_info(vid)
            row = {'date': target_date, 'variety_id': vid,
                   'variety_name': name, 'month': month}
            for i in range(1, 39):
                row[f'dim{i}'] = '-1.0000'
            writer.writerow(row)
    
    print(f'  ✅ {NUM_VARIETIES}品种CSV模板已创建（38维CA评分待填充）')
    return NUM_VARIETIES


# ============================================================
# Step 1: 回填昨日y值（与run_daily_pipeline主流程Step 1对应）
# ============================================================
def update_historical_y(data_dir: str) -> Tuple[int, int]:
    """用AKShare真实涨跌更新historical_samples.csv中缺失y值的样本
    
    返回: (更新数, 总样本数)
    """
    print(f'\n[Step 1] 回填昨日y值')
    
    csv_path = os.path.join(data_dir, 'historical_samples.csv')
    if not os.path.exists(csv_path):
        print('  ⚠️ historical_samples.csv不存在，将创建新文件')
        return 0, 0
    
    # 读取现有样本
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            samples.append(row)
    
    if not samples:
        return 0, 0
    
    cache = get_kline_cache()
    updated = 0
    
    for row in samples:
        date_str = row['date']
        vid = int(row['variety_id'])
        raw_change = float(row.get('raw_change', '0.0'))
        
        # 只更新raw_change=0（缺失真实y值）的样本
        if abs(raw_change) > 1e-8:
            continue
        
        if vid not in cache:
            continue
        
        df = cache[vid]
        target_dt = pd.to_datetime(date_str)
        mask = df['date_col'] >= target_dt
        subset = df[mask].head(2)
        
        if len(subset) >= 2:
            today_close = float(subset.iloc[0]['close'])
            next_close = float(subset.iloc[1]['close'])
            ret = (next_close - today_close) / today_close
            y = 1.0 / (1.0 + np.exp(-ret * 10))  # gain=10: 与reconstruct_samples.py一致
            
            row['y'] = f'{y:.6f}'
            row['raw_change'] = f'{ret:.6f}'
            updated += 1
    
    # 写回
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    
    print(f'  ✅ 更新了 {updated} 条y值, 总样本 {len(samples)}')
    return updated, len(samples)


# ============================================================
# Step 5: 追加今日样本
# ============================================================
def append_today_samples(date_str: str, scores_dir: str, data_dir: str) -> int:
    """将今日有真实CA评分的样本追加到historical_samples.csv
    
    只追加dim1-dim14不包含-1的行（即已有真实CA评分的品种）。
    如果今日尚未有CA评分（dim=-1），跳过追加，等CA评分写入后再追加。
    """
    print(f'\n[Step 4a] 追加今日样本')
    
    scores_csv = os.path.join(scores_dir, f'scores_{date_str}.csv')
    hist_csv = os.path.join(data_dir, 'historical_samples.csv')
    
    if not os.path.exists(scores_csv):
        print(f'  ⚠️ 今日评分文件不存在: {scores_csv}')
        return 0
    
    # 读取今日评分
    today_rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 检查是否有真实CA评分（dim值不为-1）
            dim1 = float(row.get('dim1', '-1'))
            if dim1 < 0:
                continue  # 跳过未填充CA的行
            today_rows.append(row)
    
    if not today_rows:
        print(f'  ⚠️ 今日尚无真实CA评分，跳过追加（等CA评分写入后重新运行）')
        return 0
    
    # 今日没有次日数据，y值用0.5中性占位（raw_change=0标记为待回填）
    # ⚠️ 铁律：禁止用任何公式推算y值，0.5是唯一的合法占位值
    new_samples = []
    for row in today_rows:
        dim_scores = [float(row[f'dim{i}']) for i in range(1, 39)]

        new_samples.append({
            'date': row['date'],
            'variety_id': row['variety_id'],
            'variety_name': row['variety_name'],
            'month': row['month'],
            **{f'dim{i}': row[f'dim{i}'] for i in range(1, 39)},
            'y': '0.500000',  # 中性占位，次日回填真实涨跌
            'raw_change': '0.000000',  # 标记待回填
        })
    
    # 追加写入
    fieldnames = ['date','variety_id','variety_name','month'] + \
                 [f'dim{i}' for i in range(1,39)] + \
                 ['y','raw_change']
    
    existing = []
    if os.path.exists(hist_csv):
        with open(hist_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)
    
    today_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    existing_dates = set(r['date'] for r in existing)
    if today_date in existing_dates:
        existing_today = [r for r in existing if r['date'] == today_date]
        # P2修复：检查已有行是否全为dim=-1占位。若是→替换；若已有真实CA→跳过
        existing_has_ca = any(float(r.get('dim1', '-1')) >= 0 for r in existing_today)
        if existing_has_ca:
            print(f'  ⚠️ {today_date} 已有真实CA数据({len(existing_today)}条)，跳过追加')
            return len(existing)
        else:
            print(f'  🔄 {today_date} 存在{len(existing_today)}条占位数据(dim=-1)，替换为真实CA')
            existing = [r for r in existing if r['date'] != today_date]
    
    all_rows = existing + new_samples
    
    with open(hist_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f'  ✅ 追加 {len(new_samples)} 条真实CA样本, 总样本 {len(all_rows)}')
    return len(all_rows)


# ============================================================
# Step 2: 微调
# ============================================================
def finetune_model(data_dir: str) -> Optional[NumpyCANNModel]:
    """微调CANN模型"""
    print(f'\n[Step 2] 微调模型')
    model = run_daily_training_numpy(data_dir, verbose=True)
    return model


# ============================================================
# Step 6: 全品种推理
# ============================================================
def run_inference(model: NumpyCANNModel, date_str: str, scores_dir: str, data_dir: str) -> dict:
    """全品种推理并保存结果（v4.0: 38维CA评分统一输入）
    
    安全规则：CA评分(dim1-dim38)任一为-1（未填充）的品种，跳过CANN推理，
    输出cann_score=0.5（中性），direction='无CA数据'。
    """
    print(f'\n[Step 4b] 全品种推理')
    
    scores_csv = os.path.join(scores_dir, f'scores_{date_str}.csv')
    results = []
    ca_missing = 0
    
    with open(scores_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row['variety_id'])
            name = row['variety_name']
            month = int(row['month'])
            # 读取38维CA评分（dim1-dim38）
            dim_scores = [float(row[f'dim{i}']) for i in range(1, 39)]
            
            # 任何维度=-1则跳过CANN推理，输出中性0.5
            if any(d < 0 for d in dim_scores):
                ca_missing += 1
                ca_mean = -1.0
                results.append({
                    'variety_id': vid, 'variety_name': name,
                    'ca_mean': ca_mean,
                    'cann_score': 0.5, 'direction': '无CA数据',
                })
                continue
            
            cann_score = predict_single(model, dim_scores, month, vid)
            ca_mean = np.mean(dim_scores)
            
            results.append({
                'variety_id': vid, 'variety_name': name,
                'ca_mean': ca_mean,
                'cann_score': cann_score, 'direction': '',  # 方向稍后统一判定
            })
    
    if ca_missing > 0:
        missing_names = [r['variety_name'] for r in results if r['ca_mean'] < 0]
        print(f'  ⚠️ {ca_missing}个品种CA评分未填充: {", ".join(missing_names)}，跳过推理（输出中性0.5）')
    
    # v4.1→v4.2: 回退为固定阈值 — 市场一边倒本身就是有效信号，不应强行拉平分布
    cann_scores = [r['cann_score'] for r in results]
    BULL_THRESHOLD = 0.55
    BEAR_THRESHOLD = 0.45
    
    for r in results:
        if r['ca_mean'] < 0:
            r['direction'] = '无CA数据'
        elif r['cann_score'] >= BULL_THRESHOLD:
            r['direction'] = '偏多'
        elif r['cann_score'] < BEAR_THRESHOLD:
            r['direction'] = '偏空'
        else:
            r['direction'] = '中性'
    
    bullish = [r for r in results if r['direction'] == '偏多']
    bearish = [r for r in results if r['direction'] == '偏空']
    neutral = [r for r in results if r['direction'] == '中性']
    
    # 保存JSON
    output = {
        'date': f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
        'total': len(results),
        'bullish': len(bullish), 'neutral': len(neutral), 'bearish': len(bearish),
        'cann_range': [float(min(cann_scores)), float(max(cann_scores))],
        'cann_std': float(np.std(cann_scores)),
        'cann_mean': float(np.mean(cann_scores)),
        'threshold_bull': BULL_THRESHOLD,
        'threshold_bear': BEAR_THRESHOLD,
        'scores': [{
            'variety_id': r['variety_id'],
            'variety_name': r['variety_name'],
            'ca_mean': round(r['ca_mean'], 4),
            'cann_score': round(r['cann_score'], 4),
            'direction': r['direction'],
        } for r in results]
    }
    
    json_path = os.path.join(scores_dir, f'cann_results_{date_str}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f'  ✅ 推理完成: {len(results)}品种')
    print(f'     CANN范围: [{min(cann_scores):.4f}, {max(cann_scores):.4f}], μ={np.mean(cann_scores):.4f}, σ={np.std(cann_scores):.4f}')
    print(f'     固定阈值: 偏多≥{BULL_THRESHOLD}, 偏空<{BEAR_THRESHOLD}')
    print(f'     偏多:{len(bullish)} 中性:{len(neutral)} 偏空:{len(bearish)}')
    
    if bullish:
        print(f'     偏多品种: {", ".join(r["variety_name"] for r in sorted(bullish, key=lambda x: x["cann_score"], reverse=True))}')
    
    return output


# ============================================================
# CA评分写入接口（供调度session调用CA技能后写入真实评分）
# ============================================================
def write_ca_scores(date_str: str, variety_id: int, ca_scores: List[float], scores_dir: str):
    """将CA技能输出的38维评分写入daily_scores CSV（v4.0兼容接口）

    Args:
        date_str: 日期 YYYYMMDD
        variety_id: 品种ID (0-52)
        ca_scores: 38维CA评分 [0,1]
        scores_dir: daily_scores目录

    Raises:
        ValueError: 输入参数不合法时
    """
    # 输入校验
    if not isinstance(variety_id, int) or variety_id < 0 or variety_id >= NUM_VARIETIES:
        raise ValueError(f"品种ID必须在0-{NUM_VARIETIES-1}范围内，当前={variety_id}")
    if not isinstance(ca_scores, (list, tuple)) or len(ca_scores) != 38:
        raise ValueError(f"CA评分必须为38维列表，当前维度={len(ca_scores) if isinstance(ca_scores, (list, tuple)) else type(ca_scores).__name__}")
    for i, s in enumerate(ca_scores):
        if not isinstance(s, (int, float)) or s < 0 or s > 1:
            raise ValueError(f"CA评分dim{i+1}必须在[0,1]范围内，当前={s}")
    
    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    if not os.path.exists(csv_path):
        print(f'  ⚠️ CSV不存在: {csv_path}')
        return False
    
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row['variety_id']) == variety_id:
                for i, s in enumerate(ca_scores, 1):
                    row[f'dim{i}'] = f'{s:.4f}'
            rows.append(row)
    
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return True


def check_ca_filled(date_str: str, scores_dir: str) -> Tuple[int, int]:
    """检查某日有多少品种已填充真实CA评分
    
    Returns: (已填充数, 总品种数)
    """
    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    if not os.path.exists(csv_path):
        return 0, 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    filled = sum(1 for r in rows if float(r.get('dim1', '-1')) >= 0)
    return filled, len(rows)


# ============================================================
# Step 0: Web新闻预取（供ca_scorer D2/D3/D6品种级新闻评分使用）
# ============================================================
def prefetch_web_news(data_dir: str) -> bool:
    """从Bing搜索中国商品期货新闻，存入 web_news.txt
    
    TTL=1小时，与ca_scorer缓存策略一致。
    搜索失败时留空文件（ca_scorer降级为仅SHMET）。
    
    Returns: True=预取成功, False=失败(降级)
    """
    web_file = os.path.join(data_dir, 'web_news.txt')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1小时内缓存有效
    if os.path.exists(web_file):
        mtime = os.path.getmtime(web_file)
        if time.time() - mtime < 3600 and os.path.getsize(web_file) > 100:
            print(f'  [Step 0] web_news.txt 缓存有效 ({(time.time()-mtime)/60:.0f}分钟前)')
            return True
    
    print(f'  [Step 0] 预取Web新闻...')
    
    try:
        # Bing搜索（无需API Key，HTML结果页解析）
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
                    # 提取搜索结果块
                    blocks = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)
                    for blk in blocks:
                        # 提取标题+h2/p/a中的文本
                        texts = re.findall(r'<(?:h2|p|a)[^>]*>(.*?)</(?:h2|p|a)>', blk, re.DOTALL)
                        for t in texts:
                            clean = re.sub(r'<[^>]+>', '', t).strip()
                            clean = re.sub(r'&[a-z]+;', ' ', clean)
                            clean = re.sub(r'\s+', ' ', clean)
                            if len(clean) > 20:
                                all_snippets.append(clean)
            except Exception:
                continue
        
        if all_snippets:
            # 去重（按前60字符）
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
            with open(web_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f'  [Step 0] ✅ web_news.txt 已写入 ({len(content)}字, {len(unique)}条摘要)')
            return True
        else:
            with open(web_file, 'w', encoding='utf-8') as f:
                f.write('')
            print(f'  [Step 0] ⚠️ Bing搜索无结果，降级为仅SHMET新闻')
            return False
            
    except Exception as e:
        logger.warning(f'prefetch_web_news失败: {e}')
        try:
            with open(web_file, 'w', encoding='utf-8') as f:
                f.write('')
        except:
            pass
        print(f'  [Step 0] ❌ Web搜索异常: {e}')
        return False


# ============================================================
# CA自动评分（集成ca_scorer）
# ============================================================
def _run_ca_scoring(date_str: str, scores_dir: str, data_dir: str) -> int:
    """运行CA自动评分并写入daily_scores CSV
    
    数据源降级链：AKShare → TQSDK → Sina nf_ → Web Search → None
    宏观D1-D6从缓存读取（无缓存时标记不可用），产业D7-D14从AKShare K线计算
    
    Returns: 已填充品种数
    """
    print(f'\n[Step 3b] CA自动评分 (ca_scorer)')
    
    month = int(date_str[4:6])
    all_scores, summary = score_all_varieties(data_dir, month=month, verbose=True)
    
    filled = write_all_ca_scores(all_scores, date_str, scores_dir)
    
    if summary['macro_available']:
        print(f'  ✅ CA评分已写入: {filled}/{NUM_VARIETIES}品种 (D1-D6:缓存, D7-D14:AKShare={summary["akshare_ok"]})')
    else:
        print(f'  ⚠️ CA评分部分写入: {filled}/{NUM_VARIETIES}品种 (D1-D6不可用: 无宏观缓存)')
    
    return filled


# ============================================================
# 主流程
# ============================================================
def run_daily_pipeline(data_dir: str = './CANN/data', date_str: str = None, skip_finetune: bool = False):
    """执行完整每日管线"""
    start_time = time.time()
    
    if date_str is None:
        date_str = datetime.now().strftime('%Y%m%d')
    
    # P0修复: 每次管线运行重置K线缓存，确保数据时效性
    global _kline_cache
    _kline_cache = None
    
    scores_dir = os.path.join(data_dir, 'daily_scores')
    os.makedirs(scores_dir, exist_ok=True)
    
    print('='*60)
    print(f'CANN每日管线 - {date_str}')
    print('='*60)
    
    # ============================================================
    # Step 0: 预取Web新闻（ca_scorer D2/D3/D6品种级新闻评分依赖）
    # Bing搜索 → CANN/data/web_news.txt，TTL=1小时
    # ============================================================
    prefetch_web_news(data_dir)
    
    # ============================================================
    # Step 1: 回填昨日y值（先于今日数据采集，确保微调用已标记样本）
    # ============================================================
    updated, total = update_historical_y(data_dir)
    
    # ============================================================
    # Step 2: 微调模型（仅用已标记样本：真实CA+真实y）
    # 今日新样本(y=0.5占位)不参与，等次日回填真实y后再参与训练
    # CANN核心目标：预测涨跌，CA逼近只是bootstrap
    # ============================================================
    if skip_finetune:
        print('\n[Step 2] 微调已跳过（--skip-finetune）')
        model, _ = load_pretrained_model(data_dir)
    else:
        model = finetune_model(data_dir)
        if model is None:
            model, _ = load_pretrained_model(data_dir)
    if model is None:
        print('  ⚠️ 无可用模型权重，推理结果全部为0.5（中性）')
        model = None
    
    # ============================================================
    # Step 3: CA全维度评分 — ca_scorer v3.0引擎（含38维基本面+技术面）
    # 技术面已在CA引擎内部计算，无需独立采集步骤
    # 数据源降级链：AKShare → TQSDK → Sina nf_ → Web Search → None
    # ============================================================
    n = generate_today_scores(date_str, scores_dir)
    if n == 0:
        print('❌ 无评分数据，可能是非交易日')
        return None
    _run_ca_scoring(date_str, scores_dir, data_dir)
    
    # ============================================================
    # Step 4: 追加今日样本 + 全品种推理
    # y=0.5占位, raw_change=0标记待次日回填
    # ============================================================
    total_samples = append_today_samples(date_str, scores_dir, data_dir)
    results = run_inference(model, date_str, scores_dir, data_dir)
    
    elapsed = time.time() - start_time
    print(f'\n{"="*60}')
    print(f'管线完成 - 耗时 {elapsed:.0f}s')
    print(f'总样本: {total_samples}, 今日y值待次日回填')
    print(f'{"="*60}')
    
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CANN每日管线')
    parser.add_argument('--date', default=None)
    parser.add_argument('--data-dir', default='./CANN/data')
    parser.add_argument('--skip-finetune', action='store_true', default=False,
                        help='跳过微调，直接使用预训练权重（调试用）')
    args = parser.parse_args()
    
    run_daily_pipeline(args.data_dir, args.date, args.skip_finetune)
