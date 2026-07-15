import argparse

from grading_attack import GradingAttack
from utils.config_utils import parse_config


def get_args():
    parser = argparse.ArgumentParser(description="GradingAttack: Attack & Defense Pipeline")
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run attack+defense pipeline (reads 'defenses' from config)")
    args = parser.parse_args()
    return args


def main(args):
    config = parse_config(args.config_path)
    if args.pipeline:
        config.pipeline_mode = True
    grading_attack = GradingAttack(config)
    grading_attack.run()


if __name__ == "__main__":
    args = get_args()
    main(args)
