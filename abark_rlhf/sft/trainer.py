"""
SFT Trainer for AbArk RLHF Pipeline.
Supervised Fine-Tuning — Step 1 of the RLHF workflow.
Produces a base policy model before reward training or DPO.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Config for SFT training."""
    base_model: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "./outputs/sft_model"
    dataset_path: str = "data/sft_data.jsonl"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    val_ratio: float = 0.05
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    packing: bool = True
    use_wandb: bool = False
    wandb_project: str = "abark-rlhf"


class SFTModelTrainer:
    """
    Supervised Fine-Tuning trainer.
    Step 1 in the RLHF pipeline.

    Usage:
        config = SFTConfig(base_model="Qwen/Qwen2.5-1.5B")
        trainer = SFTModelTrainer(config)
        trainer.train()
    """

    def __init__(self, config: SFTConfig):
        self.config = config

    def train(self):
        logger.info("=" * 60)
        logger.info("AbArk RLHF — SFT Training")
        logger.info("=" * 60)
        logger.info(f"Base model:  {self.config.base_model}")
        logger.info(f"Dataset:     {self.config.dataset_path}")
        logger.info(f"Max seq len: {self.config.max_seq_length}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTTrainer, SFTConfig as HF_SFTConfig
            from peft import LoraConfig, TaskType
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(f"Missing: {e}\nInstall: pip install trl peft transformers datasets accelerate")

        from ..data.dataset_utils import load_sft_jsonl, train_val_split, format_sft_for_hf

        # ── Load Data ──────────────────────────────────────────────────────
        logger.info("[Step 1/5] Loading SFT dataset...")
        samples = load_sft_jsonl(self.config.dataset_path)
        train_s, val_s = train_val_split(samples, self.config.val_ratio, self.config.seed)
        train_data = Dataset.from_list(format_sft_for_hf(train_s))
        val_data = Dataset.from_list(format_sft_for_hf(val_s))
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        # ── Load Tokenizer ─────────────────────────────────────────────────
        logger.info("[Step 2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── LoRA Config ────────────────────────────────────────────────────
        peft_config = None
        if self.config.use_lora:
            logger.info("[Step 3/5] Configuring LoRA...")
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )

        # ── Training Args ──────────────────────────────────────────────────
        logger.info("[Step 4/5] Setting training arguments...")
        training_args = HF_SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_seq_length=self.config.max_seq_length,
            packing=self.config.packing,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
        )

        # ── Train ──────────────────────────────────────────────────────────
        logger.info("[Step 5/5] Starting SFT training...")
        trainer = SFTTrainer(
            model=self.config.base_model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"SFT model saved to: {self.config.output_dir}")
