"""
DPO Trainer for AbArk RLHF Pipeline.
Direct Preference Optimization — trains a policy model directly from preference pairs.
No reward model needed.

Reference: HuggingFace TRL DPOTrainer
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """Config for DPO training."""
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir: str = "./outputs/dpo_model"
    dataset_path: str = "data/preferences.jsonl"
    beta: float = 0.1              # KL penalty coefficient
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    max_length: int = 512
    max_prompt_length: int = 256
    val_ratio: float = 0.1
    seed: int = 42
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    use_wandb: bool = False
    wandb_project: str = "abark-rlhf"


class DPOTrainer:
    """
    Direct Preference Optimization trainer.

    Requires: transformers, trl>=0.8, peft, torch, datasets
    Install: pip install trl peft transformers datasets accelerate

    Usage:
        config = DPOConfig(base_model="mistralai/Mistral-7B-Instruct-v0.2")
        trainer = DPOTrainer(config)
        trainer.train()
    """

    def __init__(self, config: DPOConfig):
        self.config = config

    def train(self):
        logger.info("=" * 60)
        logger.info("AbArk RLHF — DPO Training")
        logger.info("=" * 60)
        logger.info(f"Base model:  {self.config.base_model}")
        logger.info(f"Beta (KL):   {self.config.beta}")
        logger.info(f"Dataset:     {self.config.dataset_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import DPOTrainer as HF_DPOTrainer, DPOConfig as HF_DPOConfig
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}\nInstall: pip install trl peft transformers datasets accelerate")

        from ..data.dataset_utils import load_preference_jsonl, train_val_split, format_preference_for_hf

        # ── Load Data ──────────────────────────────────────────────────────
        logger.info("[Step 1/5] Loading preference dataset...")
        samples = load_preference_jsonl(self.config.dataset_path)
        train_samples, val_samples = train_val_split(samples, self.config.val_ratio, self.config.seed)
        train_data = Dataset.from_list(format_preference_for_hf(train_samples))
        val_data = Dataset.from_list(format_preference_for_hf(val_samples))
        logger.info(f"  Train: {len(train_data)} | Val: {len(val_data)}")

        # ── Load Tokenizer + Model ─────────────────────────────────────────
        logger.info("[Step 2/5] Loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

        if self.config.use_lora:
            logger.info("[Step 3/5] Applying LoRA...")
            lora_cfg = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                task_type=TaskType.CAUSAL_LM,
                bias="none",
                target_modules=["q_proj", "v_proj"],
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()

        # ── Training Args ──────────────────────────────────────────────────
        logger.info("[Step 4/5] Configuring DPO training arguments...")
        training_args = HF_DPOConfig(
            output_dir=self.config.output_dir,
            beta=self.config.beta,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            logging_steps=10,
            bf16=torch.cuda.is_available(),
        )

        # ── Train ──────────────────────────────────────────────────────────
        logger.info("[Step 5/5] Starting DPO training...")
        trainer = HF_DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            tokenizer=tokenizer,
        )
        trainer.train()
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        logger.info(f"DPO model saved to: {self.config.output_dir}")
