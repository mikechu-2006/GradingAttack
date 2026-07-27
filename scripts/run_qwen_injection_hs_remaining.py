#!/usr/bin/env python3
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grading_attack import GradingAttack
from utils.config_utils import parse_config
from utils.data_utils import read_student_qa_data_from_jsonl


def completed_indices(result_glob):
    done = set()
    for path in Path(".").glob(result_glob):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(json.loads(line).get("source_index"))
    return done


def write_dataset(source_path, indices, output_path):
    rows = read_student_qa_data_from_jsonl(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx in indices:
            row = rows[idx].as_dict()
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    config_path, model_key, shard_id, shards = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    config = parse_config(config_path)
    config.pipeline_mode = True
    source_path = config.data_config[0].path
    total_rows = len(read_student_qa_data_from_jsonl(source_path))
    all_target = list(range(50))

    if model_key == "qwen3":
        glob = "result_Injection-Qwen3-4B-Instruct-2507-hijacking-suppression-50/roleplay/Qwen3-4B-Instruct-2507/*.jsonl"
    else:
        glob = "result_Injection-Qwen3.5-4B-hijacking-suppression-50/roleplay/Qwen3.5-4B/*.jsonl"

    done = completed_indices(glob)
    remaining = [idx for idx in all_target if idx not in done and idx < total_rows]
    shard_indices = remaining[shard_id::shards]

    config = copy.deepcopy(config)
    config.name = f"{config.name}-remaining-shard{shard_id}"
    config.log_config.log_dir = f"{config.log_config.log_dir}_remaining_shard{shard_id}"
    config.log_config.result_dir = f"{config.log_config.result_dir}_remaining_shard{shard_id}"
    tmp_dataset = Path("/tmp/grading_attack_qwen_injection_hs_remaining") / f"{model_key}_shard{shard_id}.jsonl"
    write_dataset(source_path, shard_indices, tmp_dataset)
    config.data_config[0].path = str(tmp_dataset)
    config.data_config[0].max_samples = None
    config.data_config[0].random_seed = None
    config.data_config[0].sample_offset = 0

    if not shard_indices:
        print(f"{model_key} shard{shard_id}: no remaining samples")
        return
    print(f"{model_key} shard{shard_id}: {len(shard_indices)} samples {shard_indices}", flush=True)
    GradingAttack(config).run()


if __name__ == "__main__":
    main()
