#!/bin/bash
# 兼容入口：Hijacking Suppression 网格搜索
export DEFENSE_TYPE=hijacking_suppression
exec bash scripts/run_roleplay_defense_grid_search.sh
