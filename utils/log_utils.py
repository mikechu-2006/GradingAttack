import os
import json
import logging
from typing import Dict
from datetime import datetime

from utils.config_utils import AttackConfig
from utils.experiment_paths import experiment_log_dir, experiment_result_dir


class GradingAttackLogger:
    def __init__(self, config: AttackConfig):
        time_string = datetime.now().strftime("%Y%m%d%H%M")
        log_root = experiment_log_dir(config)
        result_root = experiment_result_dir(config)
        self.log_path = os.path.join(
            log_root,
            f"{config.name}_{time_string}.log",
        )
        self.result_path = os.path.join(
            result_root,
            f"{config.name}_{time_string}.jsonl",
        )

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        self.logger = logging.getLogger()
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setLevel(logging.DEBUG)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def info(self, s: str):
        self.logger.info(s)

    def result(self, r: Dict):
        with open(self.result_path, "a", encoding="utf-8") as resule_file:
            resule_file.write(json.dumps(r, ensure_ascii=False) + "\n")

    @property
    def metrics_path(self):
        return self.result_path.replace(".jsonl", "_metrics.json")
    