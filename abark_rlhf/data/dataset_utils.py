"""
Data utilities for AbArk RLHF Pipeline.
Handles preference dataset loading, formatting, and splits.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PreferenceSample:
    """A single preference pair: one chosen, one rejected response."""
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SFTSample:
    """A supervised fine-tuning sample."""
    prompt: str
    response: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def load_preference_jsonl(path: str) -> List[PreferenceSample]:
    """
    Load preference pairs from a JSONL file.
    Each line: {"prompt": ..., "chosen": ..., "rejected": ...}
    """
    samples = []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(PreferenceSample(
                    prompt=obj["prompt"],
                    chosen=obj["chosen"],
                    rejected=obj["rejected"],
                    metadata=obj.get("metadata", {}),
                ))
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping malformed line {i+1}: {e}")

    logger.info(f"Loaded {len(samples)} preference samples from {path}")

    # Log first 3 samples
    for i, s in enumerate(samples[:3]):
        logger.debug(f"  Sample {i+1}: prompt='{s.prompt[:60]}...'")

    return samples


def load_sft_jsonl(path: str) -> List[SFTSample]:
    """
    Load SFT samples from a JSONL file.
    Each line: {"prompt": ..., "response": ...}
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                samples.append(SFTSample(
                    prompt=obj["prompt"],
                    response=obj["response"],
                    metadata=obj.get("metadata", {}),
                ))
            except (KeyError, json.JSONDecodeError) as e:
                logger.warning(f"Skipping malformed line {i+1}: {e}")

    logger.info(f"Loaded {len(samples)} SFT samples from {path}")
    return samples


def train_val_split(
    samples: list,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[list, list]:
    """Split dataset into train and validation sets."""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * (1 - val_ratio))
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    logger.info(f"Split: {len(train)} train / {len(val)} val")
    return train, val


def format_preference_for_hf(samples: List[PreferenceSample]) -> List[Dict]:
    """
    Convert PreferenceSample list to HuggingFace TRL format.
    Compatible with trl.RewardTrainer and trl.DPOTrainer.
    """
    return [
        {
            "prompt": s.prompt,
            "chosen": s.chosen,
            "rejected": s.rejected,
        }
        for s in samples
    ]


def format_sft_for_hf(samples: List[SFTSample]) -> List[Dict]:
    """
    Convert SFTSample list to HuggingFace TRL SFT format.
    Uses chat template: [{"role": "user", ...}, {"role": "assistant", ...}]
    """
    return [
        {
            "messages": [
                {"role": "user", "content": s.prompt},
                {"role": "assistant", "content": s.response},
            ]
        }
        for s in samples
    ]


def create_dummy_preference_dataset(n: int = 100, path: str = "data/dummy_preferences.jsonl"):
    """Create a small synthetic preference dataset for testing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(n):
        samples.append({
            "prompt": f"Question {i}: What is the best approach to task {i}?",
            "chosen": f"A thoughtful, detailed answer to question {i} with clear reasoning and examples.",
            "rejected": f"A vague, unhelpful answer to {i} without any detail.",
        })
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"Created dummy dataset: {path} ({n} samples)")
    return path
