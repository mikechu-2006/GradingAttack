#!/bin/bash
# 兼容入口：Attention Sharpening 网格搜索
export DEFENSE_TYPE=attention_sharpening
exec bash scripts/run_roleplay_defense_grid_search.sh
