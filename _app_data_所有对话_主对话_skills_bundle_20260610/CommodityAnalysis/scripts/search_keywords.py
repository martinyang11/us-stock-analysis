#!/usr/bin/env python3
"""
商品期货基本面分析 - 搜索关键词生成器
根据给定品种，生成14维度所需的搜索关键词列表
"""

import json
from datetime import datetime, timedelta, timezone

# P1-12: 品种类别映射（补全缺失品种）
CATEGORY_MAP = {
    # 黑色系
    "螺纹钢": "黑色系", "热卷": "黑色系", "铁矿石": "黑色系",
    "焦煤": "黑色系", "焦炭": "黑色系", "硅铁": "黑色系",
    "锰硅": "黑色系", "氧化铝": "黑色系", "工业硅": "黑色系",
    # 有色金属
    "沪铜": "有色金属", "沪铝": "有色金属", "沪锌": "有色金属",
    "沪镍": "有色金属", "沪锡": "有色金属",
    # 贵金属
    "黄金": "贵金属", "白银": "贵金属",
    # 能化
    "原油": "能化", "燃油": "能化", "沥青": "能化", "低硫燃油": "能化",
    "PTA": "能化", "甲醇": "能化", "PP": "能化", "塑料": "能化",
    "PVC": "能化", "橡胶": "能化", "20号胶": "能化",
    "纯碱": "能化", "玻璃": "能化", "乙二醇": "能化",
    "短纤": "能化", "苯乙烯": "能化", "纸浆": "能化",
    "尿素": "能化", "碳酸锂": "能化",
    # 农产品
    "豆一": "农产品", "豆二": "农产品", "豆粕": "农产品",
    "豆油": "农产品", "棕榈油": "农产品", "菜油": "农产品",
    "菜粕": "农产品", "玉米": "农产品", "淀粉": "农产品",
    "生猪": "农产品", "鸡蛋": "农产品",
    "棉花": "农产品", "棉纱": "农产品",
    "白糖": "农产品", "苹果": "农产品", "红枣": "农产品",
    "花生": "农产品", "集运指数(欧线)": "农产品",
}

# 外盘映射
OVERSEAS_MAP = {
    "沪铜": ["LME铜", "COMEX铜"], "沪铝": ["LME铝"], "沪锌": ["LME锌"],
    "沪镍": ["LME镍"], "沪锡": ["LME锡"], "黄金": ["COMEX黄金"],
    "白银": ["COMEX白银"], "原油": ["WTI原油", "布伦特原油"],
    "豆粕": ["CBOT大豆"], "豆油": ["CBOT豆油"], "棕榈油": ["BMD棕榈油"],
    "玉米": ["CBOT玉米"], "铁矿石": ["新加坡铁矿石"],
}


def get_time_window():
    """获取当前UTC时间和4小时窗口"""
    now = datetime.now(timezone.utc)
    window_end = now
    window_start = now - timedelta(hours=4)
    return now, window_start, window_end


def generate_search_keywords(variety: str) -> dict:
    """生成14维度的搜索关键词"""
    category = CATEGORY_MAP.get(variety, "通用")
    overseas = OVERSEAS_MAP.get(variety, [])

    now, window_start, window_end = get_time_window()

    keywords = {
        "时间基准": {
            "当前UTC时间": now.strftime("%Y-%m-%d %H:%M UTC"),
            "窗口起始": window_start.strftime("%Y-%m-%d %H:%M UTC"),
            "窗口结束": window_end.strftime("%Y-%m-%d %H:%M UTC"),
        },
        "1_货币政策": [
            f"美联储 利率决议 最新",
            f"人民银行 MLF 逆回购 LPR 最新",
            f"美元指数 人民币汇率 最新",
            f"欧央行 利率 {variety}",
        ],
        "2_地缘政治": [
            f"地缘冲突 中东 俄乌 最新",
            f"贸易摩擦 制裁 {variety}",
            f"战争 停火 最新消息",
        ],
        "3_产业政策": [
            f"{variety} 限产 补贴 环保 最新",
            f"{variety} 产能 政策 发改委",
            f"{category} 产业政策 最新",
        ],
        "4_关键经济指标": [
            f"中国 PMI CPI PPI 最新发布",
            f"美国 非农 CPI GDP 最新",
            f"中国 M2 社融 最新",
            f"{variety} 经济指标 影响",
        ],
        "5_市场情绪": [
            f"VIX恐慌指数 最新",
            # P5-4: DXY已移除，保留在维度1的货币政策中
            f"{variety} CFTC持仓 资金流向",
            f"{variety} 持仓量 成交量",
        ],
        "6_财政政策": [
            f"专项债 基建投资 最新",
            f"特别国债 财政刺激 最新",
            f"{variety} 财政政策 影响",
        ],
        "7_供应": [
            f"{variety} 产量 开工率 最新",
            f"{variety} 库存 进口 最新",
            f"{variety} 产能 检修 最新",
        ],
        "8_需求": [
            f"{variety} 需求 消费 最新",
            f"{variety} 下游开工 表观消费",
            f"{variety} 出口 终端需求",
        ],
        "9_基差": [
            f"{variety} 基差 现货价格 期货价格",
            f"{variety} 升贴水 最新",
        ],
        "10_跨市场": [],
        "11_价格位置": [
            f"{variety} 技术分析 支撑 阻力",
            f"{variety} 均线 趋势 最新",
            f"{variety} 波动率 布林带",
        ],
        "12_周期性": [
            f"{variety} 季节性 淡旺季 周期",
            f"{variety} 检修季 库存周期",
            f"{variety} 历年同期 价格走势",
        ],
        # P5: 新增维度13 - 资金面
        "13_资金面": [
            f"{variety} 持仓量 变化 最新",
            f"{variety} 仓单 周度变化",
            f"{variety} 前20会员 持仓 多空",
            f"{variety} 沉淀资金 净流入",
        ],
        # P5: 新增维度14 - 价差结构
        "14_价差结构": [
            f"{variety} 近月 远月 跨期价差",
            f"{variety} contango backwardation 结构",
            f"{variety} 月间套利 信号",
        ],
    }

    # 跨市场搜索关键词
    if overseas:
        for mkt in overseas:
            keywords["10_跨市场"].append(f"{mkt} 价格 最新")
            keywords["10_跨市场"].append(f"{mkt} {variety} 内外盘 价差")
    else:
        keywords["10_跨市场"] = [f"{variety} 国际价格 外盘 最新"]

    return keywords


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python search_keywords.py <品种名>")
        print("示例: python search_keywords.py 沪铜")
        sys.exit(1)

    variety = sys.argv[1]
    result = generate_search_keywords(variety)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
