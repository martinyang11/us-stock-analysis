#!/bin/bash
# SANN Pipeline launchd 包装脚本 — 工作日才执行
# 由 launchd com.gtrade.sann-pipeline 调用

export HTTPS_PROXY="http://127.0.0.1:7897"

DAYOFWEEK=$(date +%u)
if [ "$DAYOFWEEK" -gt 5 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S'): 周末跳过 SANN Pipeline"
    exit 0
fi

cd /Users/martin/Projects/合约分析
exec /Users/martin/Projects/合约分析/.venv/bin/python \
    /Users/martin/Projects/合约分析/skills/SANN/scripts/daily_pipeline.py \
    --data-dir /Users/martin/Projects/合约分析/skills/SANN/data
