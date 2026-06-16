#!/usr/bin/env python3
"""
SANN每日管线脚本 — TradFi gTrade版 v4.2
每日 UTC 21:00 执行：回填昨日y值 → 微调 → 技术面采集 → 追加今日样本 → 推理

核心原则：
- SA统一输出14维基本面评分，技术面24维由yfinance K线计算
- y值来自yfinance次日真实涨跌（美股收盘价为日切）
- 只有同时具备真实SA+真实y的样本才是有效训练数据
- 今日新采集的SA数据(y=0.5占位)不参与当日微调

调度：每日 UTC 21:00 执行（美股收盘后约1小时）
"""

import os
import sys
import subprocess
import csv
import json
import time
import warnings
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ===== 环境自检 =====
REQUIRED_PKGS = ['numpy', 'pandas', 'requests']
for pkg in REQUIRED_PKGS:
    try:
        __import__(pkg)
    except ImportError:
        print(f"[自检] 安装缺失依赖: {pkg}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import pandas as pd

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# skills/SANN/scripts → skills/StockAnalysis/scripts
SA_SCRIPTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'StockAnalysis', 'scripts'))
# skills/SANN/scripts → skills/common
COMMON_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'common'))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
for p in [PROJECT_ROOT, COMMON_DIR, SA_SCRIPTS_DIR, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# 导入品种列表（统一权威源，gTrade动态加载）
from skills.common.variety_list import (
    VARIETY_CODES, VARIETY_NAMES, SYMBOLS, SYMBOL_TO_ID, NUM_VARIETIES
)

# 导入gTrade数据接口
from gtrade_data import GtradeDataProvider

logger = logging.getLogger('SANN.pipeline')


# ============================================================
# K线缓存
# ============================================================
_kline_cache = None


def get_kline_cache() -> Dict[int, pd.DataFrame]:
    """懒加载全币种K线缓存（gTrade API）"""
    global _kline_cache
    if _kline_cache is not None:
        return _kline_cache

    _kline_cache = {}
    print(f'  [K线缓存] 从gTrade获取 {NUM_VARIETIES} 币种日K线...')

    ok = 0
    with GtradeDataProvider() as provider:
        for vid in range(NUM_VARIETIES):
            symbol = VARIETY_CODES[vid]
            try:
                klines = provider.get_klines(symbol, interval="1d", limit=500)
                if klines and len(klines) >= 50:
                    df = pd.DataFrame(klines)
                    df['date_col'] = pd.to_datetime(df['open_time'], unit='ms')
                    df = df.sort_values('date_col', ascending=True).reset_index(drop=True)
                    _kline_cache[vid] = df
                    ok += 1
                else:
                    print(f'    ⚠️ {symbol}: 数据不足 ({len(klines) if klines else 0}条)')
                time.sleep(0.05)  # 频率控制
            except Exception as e:
                print(f'    ❌ {symbol}: {e}')

    failed = NUM_VARIETIES - ok
    print(f'  [K线缓存] gTrade OK={ok}/{NUM_VARIETIES}' + (f' 失败={failed}' if failed else ''))
    return _kline_cache


# ============================================================
# Step 1: 回填昨日y值
# ============================================================
def update_historical_y(data_dir: str) -> Tuple[int, int]:
    """用gTrade次日涨跌更新historical_samples.csv中缺失y值的样本"""
    print(f'\n[Step 1] 回填昨日y值')

    csv_path = os.path.join(data_dir, 'historical_samples.csv')
    if not os.path.exists(csv_path):
        print('  ⚠️ historical_samples.csv不存在')
        return 0, 0

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
        vid = int(row['variety_id'] if 'variety_id' in row else row.get('variety_id', 0))
        raw_change = float(row.get('raw_change', '0.0'))

        if abs(raw_change) > 1e-8:
            continue  # 已有真实y值

        if vid not in cache:
            continue

        df = cache[vid]
        target_dt = pd.to_datetime(date_str)
        # 找当日收盘
        mask_today = df['date_col'] >= target_dt
        subset_today = df[mask_today]
        if len(subset_today) < 1:
            continue
        today_close = float(subset_today.iloc[0]['close'])

        # 找下一个交易日收盘 (跳过周末/假期: 至少+1日历天后)
        next_dt = target_dt + pd.Timedelta(days=1)
        mask_next = df['date_col'] >= next_dt
        subset_next = df[mask_next]
        if len(subset_next) < 1:
            continue
        next_close = float(subset_next.iloc[0]['close'])

        ret = (next_close - today_close) / today_close
        y = 1.0 / (1.0 + np.exp(-ret * 20))

        row['y'] = f'{y:.6f}'
        row['raw_change'] = f'{ret:.6f}'
        updated += 1

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)

    print(f'  ✅ 更新了 {updated} 条y值, 总样本 {len(samples)}')
    return updated, len(samples)


# ============================================================
# Step 2: 微调模型（代理到 pretrain_numpy）
# ============================================================
def finetune_model(data_dir: str):
    """微调SANN模型"""
    print(f'\n[Step 2] 微调模型')

    # 延迟导入确保路径正确
    cann_dir = os.path.dirname(data_dir)  # SANN/data → SANN
    scripts_dir = os.path.join(cann_dir, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from pretrain_numpy import run_daily_training_numpy, NumpySANNModel
        model = run_daily_training_numpy(data_dir, verbose=True)
        return model
    except ImportError as e:
        print(f'  ⚠️ 无法导入pretrain_numpy: {e}')
        return None


# ============================================================
# Step 3: 创建评分模板 + 技术面自动填充
# ============================================================
def generate_today_scores(date_str: str, scores_dir: str) -> int:
    """创建今日评分CSV，自动填充24维技术面评分，D1-D14待CA填充"""
    print(f'\n[Step 3] 创建 {date_str} 评分模板 + 技术面填充')

    target_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    month = int(date_str[4:6])

    fieldnames = ['date', 'variety_id', 'variety_name', 'month'] + \
                 [f'dim{i}' for i in range(1, 15)] + \
                 [f'tech{i}' for i in range(1, 25)]

    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')

    # 已存在且有真实CA数据时跳过
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row.get('dim1', '-1')) >= 0:
                    print(f'  ⚠️ CSV已存在真实CA评分，跳过覆盖')
                    return NUM_VARIETIES
                break

    cache = get_kline_cache()

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for vid in range(NUM_VARIETIES):
            name = VARIETY_NAMES.get(vid, f"品种{vid}")
            row = {
                'date': target_date,
                'variety_id': vid,
                'variety_name': name,
                'month': month,
            }

            # D1-D14: 待CA填充 (-1)
            for i in range(1, 15):
                row[f'dim{i}'] = '-1.0000'

            # T1-T24: 技术面自动计算
            tech_filled = False
            if vid in cache:
                try:
                    df = cache[vid]
                    from gtrade_data import compute_technical_scores
                    tech_scores = compute_technical_scores(df)
                    if len(tech_scores) == 24:
                        for i, s in enumerate(tech_scores, 1):
                            row[f'tech{i}'] = f'{s:.4f}'
                        tech_filled = True
                except Exception:
                    pass

            if not tech_filled:
                for i in range(1, 25):
                    row[f'tech{i}'] = '-1.0000'

            writer.writerow(row)

    print(f'  ✅ {NUM_VARIETIES}币种CSV模板已创建（技术面自动填充）')
    return NUM_VARIETIES


# ============================================================
# Step 4: CA评分写入接口
# ============================================================
def write_ca_scores(date_str: str, variety_id: int, ca_scores: List[float],
                     scores_dir: str):
    """将CA技能输出的14维评分写入daily_scores CSV

    Args:
        date_str: 日期 YYYYMMDD
        variety_id: 币种ID (0-49)
        ca_scores: 14维CA评分 [0,1]
        scores_dir: daily_scores目录
    """
    if not isinstance(variety_id, int) or variety_id < 0 or variety_id >= NUM_VARIETIES:
        raise ValueError(f"币种ID必须在0-{NUM_VARIETIES-1}范围内")
    if len(ca_scores) != 14:
        raise ValueError(f"CA评分必须为14维列表")

    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    if not os.path.exists(csv_path):
        print(f'  ⚠️ CSV不存在: {csv_path}')
        return False

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row.get('variety_id', row.get('variety_id', -1))) == variety_id:
                for i, s in enumerate(ca_scores, 1):
                    row[f'dim{i}'] = f'{s:.4f}'
            rows.append(row)

    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return True


# ============================================================
# Step 5: 追加今日样本 + 全品种推理
# ============================================================
def append_today_samples(date_str: str, scores_dir: str, data_dir: str) -> int:
    """将有真实CA评分的今日样本追加到historical_samples.csv"""
    print(f'\n[Step 5a] 追加今日样本')

    scores_csv = os.path.join(scores_dir, f'scores_{date_str}.csv')
    hist_csv = os.path.join(data_dir, 'historical_samples.csv')

    if not os.path.exists(scores_csv):
        print(f'  ⚠️ 今日评分文件不存在')
        return 0

    today_rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim1 = float(row.get('dim1', '-1'))
            if dim1 < 0:
                continue
            today_rows.append(row)

    if not today_rows:
        print(f'  ⚠️ 今日尚无真实CA评分，跳过追加')
        return 0

    new_samples = []
    for row in today_rows:
        # 提取14维CA + 24维技术面
        dims_14 = [row[f'dim{i}'] for i in range(1, 15)]
        techs_24 = [row[f'tech{i}'] for i in range(1, 25)]

        new_samples.append({
            'date': row['date'],
            'variety_id': row.get('variety_id', row.get('variety_id', '0')),
            'variety_name': row.get('variety_name', row.get('variety_name', '')),
            'month': row['month'],
            **{f'dim{i}': row[f'dim{i}'] for i in range(1, 15)},
            **{f'tech{i}': row[f'tech{i}'] for i in range(1, 25)},
            'y': '0.500000',
            'raw_change': '0.000000',
        })

    fieldnames = ['date', 'variety_id', 'variety_name', 'month'] + \
                 [f'dim{i}' for i in range(1, 15)] + \
                 [f'tech{i}' for i in range(1, 25)] + \
                 ['y', 'raw_change']

    existing = []
    if os.path.exists(hist_csv):
        with open(hist_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.append(row)

    today_date = f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}'
    existing = [r for r in existing if r.get('date', '') != today_date]
    all_rows = existing + new_samples

    with open(hist_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'  ✅ 追加 {len(new_samples)} 条真实CA样本, 总样本 {len(all_rows)}')
    return len(all_rows)


def run_inference(model, date_str: str, scores_dir: str, data_dir: str) -> dict:
    """全币种推理"""
    print(f'\n[Step 5b] 全币种推理')

    scores_csv = os.path.join(scores_dir, f'scores_{date_str}.csv')
    results = []
    ca_missing = 0

    # 确保 pretrain_numpy 可导入
    cann_dir = os.path.dirname(data_dir)
    scripts_dir = os.path.join(cann_dir, 'scripts')
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from pretrain_numpy import predict_single
    except ImportError:
        print('  ⚠️ 无法导入 pretrain_numpy，全部输出0.5')
        model = None

    with open(scores_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row.get('variety_id', row.get('variety_id', 0)))
            name = row.get('variety_name', row.get('variety_name', ''))
            month = int(row['month'])

            dims_14 = [float(row[f'dim{i}']) for i in range(1, 15)]
            techs_24 = [float(row[f'tech{i}']) for i in range(1, 25)]

            # 任何维度=-1则跳过
            if any(d < 0 for d in dims_14) or model is None:
                ca_missing += 1
                results.append({
                    'variety_id': cid, 'variety_name': name,
                    'ca_mean': np.mean([d for d in dims_14 if d >= 0]) if any(d >= 0 for d in dims_14) else -1,
                    'cann_score': 0.5, 'direction': '无CA数据' if any(d < 0 for d in dims_14) else '无模型',
                })
                continue

            # 合并38维输入
            dims_38 = dims_14 + techs_24
            cann_score = predict_single(model, dims_38, month, cid)
            ca_mean = float(np.mean(dims_14))

            results.append({
                'variety_id': cid, 'variety_name': name,
                'ca_mean': round(ca_mean, 4),
                'cann_score': round(cann_score, 4),
                'direction': '',
            })

    # 方向判定
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

    cann_scores = [r['cann_score'] for r in results]
    bullish = [r for r in results if r['direction'] == '偏多']
    bearish = [r for r in results if r['direction'] == '偏空']
    neutral = [r for r in results if r['direction'] == '中性']

    output = {
        'date': f'{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}',
        'total': len(results),
        'bullish': len(bullish), 'neutral': len(neutral), 'bearish': len(bearish),
        'cann_range': [float(min(cann_scores)), float(max(cann_scores))],
        'cann_std': float(np.std(cann_scores)),
        'cann_mean': float(np.mean(cann_scores)),
        'scores': results,
    }

    json_path = os.path.join(scores_dir, f'cann_results_{date_str}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f'  ✅ 推理完成: {len(results)}币种')
    print(f'     SANN范围: [{min(cann_scores):.4f}, {max(cann_scores):.4f}]')
    print(f'     μ={np.mean(cann_scores):.4f} σ={np.std(cann_scores):.4f}')
    print(f'     偏多:{len(bullish)} 中性:{len(neutral)} 偏空:{len(bearish)}')

    # 显示Top5多和Top5空
    if bullish:
        top5_long = sorted(bullish, key=lambda x: x['cann_score'], reverse=True)[:5]
        print(f'     🔵 Top5偏多: {", ".join(f"{r["variety_name"]}({r["cann_score"]:.4f})" for r in top5_long)}')
    if bearish:
        top5_short = sorted(bearish, key=lambda x: x['cann_score'])[:5]
        print(f'     🔴 Top5偏空: {", ".join(f"{r["variety_name"]}({r["cann_score"]:.4f})" for r in top5_short)}')

    return output


# ============================================================
# 主流程
# ============================================================
def run_daily_pipeline(data_dir: str = './SANN/data', date_str: str = None,
                       skip_finetune: bool = False):
    """执行完整每日管线"""
    start_time = time.time()

    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')

    global _kline_cache
    _kline_cache = None

    scores_dir = os.path.join(data_dir, 'daily_scores')
    os.makedirs(scores_dir, exist_ok=True)

    print('=' * 60)
    print(f'SANN每日管线 (Crypto) - {date_str}')
    print(f'执行时间: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}')
    print('=' * 60)

    # Step 1: 回填昨日y值
    updated, total = update_historical_y(data_dir)

    # Step 2: 微调模型
    if skip_finetune:
        print('\n[Step 2] 微调已跳过')
        model = None
    else:
        model = finetune_model(data_dir)

    if model is None:
        # 尝试加载已有权重
        try:
            from pretrain_numpy import load_pretrained_model
            model, _ = load_pretrained_model(data_dir)
        except Exception:
            model = None

    if model is None:
        print('  ⚠️ 无可用模型权重，推理结果全部为0.5')

    # Step 3: 创建评分模板 + 技术面自动填充
    n = generate_today_scores(date_str, scores_dir)
    if n == 0:
        print('❌ 无评分数据')
        return None

    # Step 4 提示: CA评分需要手动或通过CA调度执行
    print(f'\n[Step 4] CA评分待填充 — 请执行CA分析后将评分写入')
    filled = check_ca_filled(date_str, scores_dir)
    print(f'  当前已填充: {filled[0]}/{filled[1]} 币种')

    # Step 5: 追加样本 + 推理
    total_samples = append_today_samples(date_str, scores_dir, data_dir)
    results = run_inference(model, date_str, scores_dir, data_dir) if total_samples > 0 else None

    elapsed = time.time() - start_time
    print(f'\n{"=" * 60}')
    print(f'管线完成 - 耗时 {elapsed:.0f}s')
    print(f'总样本: {total_samples}, 今日y值待 ETH 00:00后回填')
    print(f'{"=" * 60}')

    return results


def check_ca_filled(date_str: str, scores_dir: str) -> Tuple[int, int]:
    """检查CA评分填充进度"""
    csv_path = os.path.join(scores_dir, f'scores_{date_str}.csv')
    if not os.path.exists(csv_path):
        return 0, 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    filled = sum(1 for r in rows if float(r.get('dim1', '-1')) >= 0)
    return filled, len(rows)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SANN每日管线 (Crypto)')
    parser.add_argument('--date', default=None)
    parser.add_argument('--data-dir', default='./SANN/data')
    parser.add_argument('--skip-finetune', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')

    run_daily_pipeline(args.data_dir, args.date, args.skip_finetune)
