#!/usr/bin/env python
"""
Run SFT training.
Usage: python scripts/run_sft.py --config configs/sft_config.yaml
"""

import argparse
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from abark_rlhf.sft.trainer import SFTModelTrainer, SFTConfig


def main():
    parser = argparse.ArgumentParser(description="AbArk RLHF — SFT Training")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    # CLI overrides
    if args.base_model:
        cfg_dict["base_model"] = args.base_model
    if args.dataset_path:
        cfg_dict["dataset_path"] = args.dataset_path
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir

    config = SFTConfig(**{k: v for k, v in cfg_dict.items() if hasattr(SFTConfig, k)})
    trainer = SFTModelTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
