#!/bin/bash
export DEFENSE_TYPE="${DEFENSE_TYPE:-all}"
exec bash scripts/roleplay_defenses/run_grid_search.sh
