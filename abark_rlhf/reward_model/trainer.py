"""
Reward Model Trainer for AbArk RLHF Pipeline.
Trains a Bradley-Terry reward model using HuggingFace TRL.

Reference: RLHFlow/RLHF-Reward-Modeling
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RewardModelConfig:
    """Config for reward model training."""
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir: str = "./outputs/reward_model"
    dataset_path: str = "data/preferences.jsonl"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_length: int = 512
    val_ratio: float = 0.1
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_wandb: bool = False
    wandb_project: str = "abark-rlhf"


class RewardModelTrainer:
    """
    Trains a Bradley-Terry reward model.

    Requires: transformers, trl, peft, torch, datasets
    Install: pip install trl peft transformers datasets accelerate

    Usage:
        config = RewardModelConfig(base_model="mistralai/Mistral-7B-v0.1")
        trainer = RewardModelTrainer(config)
        trainer.train()
    """

    def __init__(self, config: RewardModelConfig):
        self.config = config

    def train(self):
        """Run the full reward model training pipeline."""
        logger.info("=" * 60)
        logger.info("AbArk RLHF — Reward Model Training")
        logger.info("=" * 60)
        logger.info(f"Base model:    {self.config.base_model}")
        logger.info(f"Dataset:       {self.config.dataset_path}")
        logger.info(f"Output dir:    {self.config.output_dir}")
        logger.info(f"Epochs:        {self.config.num_train_epochs}")
        logger.info(f"LoRA enabled:  {self.config.use_lora}")

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
            from trl import RewardTrainer, RewardConfig
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}\n"
                "Install with: pip install trl peft transformers datasets accelerate"
            )

        from ..data.dataset_utils import load_preference_jsonl, train_val_split, format_preference_for_hf

        # ── Step 1: Load Data ─────────────────────────────────────────────
        logger.info("[Step 1/5] Loading preference dataset...")
        samples = load_preference_jsonl(self.config.dataset_path)
        train_samples, val_samples = train_val_split(samples, self.config.val_ratio, self.config.seed)
        train_data = Dataset.from_list(format_preference_for_hf(train_samples))
        val_data = Dataset.from_list(format_preference_for_hf(val_samples))
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        # ── Step 2: Load Tokenizer ────────────────────────────────────────
        logger.info("[Step 2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"  Tokenizer vocab size: {tokenizer.vocab_size}")

        # ── Step 3: Load Model ────────────────────────────────────────────
        logger.info("[Step 3/5] Loading base model for reward modeling...")
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        if self.config.use_lora:
            logger.info("  Applying LoRA adapters...")
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.SEQ_CLS,
                bias="none",
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # ── Step 4: Training Arguments ────────────────────────────────────
        logger.info("[Step 4/5] Setting up training arguments...")
        training_args = RewardConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_length,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
        )

        # ── Step 5: Train ─────────────────────────────────────────────────
        logger.info("[Step 5/5] Starting reward model training...")
        trainer = RewardTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"Reward model saved to: {self.config.output_dir}")
