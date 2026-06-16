#!/bin/bash
# CatTrader launchd 包装脚本 — 工作日才执行
# 由 launchd com.gtrade.cattrader 调用

export HTTPS_PROXY="http://127.0.0.1:7897"

DAYOFWEEK=$(date +%u)
if [ "$DAYOFWEEK" -gt 5 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'): 周末跳过 CatTrader"
    exit 0
fi

cd /Users/martin/Projects/合约分析
exec /Users/martin/Projects/合约分析/.venv/bin/python \
    /Users/martin/Projects/合约分析/skills/CatTrader/scripts/cattrader.py
