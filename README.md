# abark-rlhf-pipeline

> **RLHF Training Toolkit by AbArk**
> End-to-end Reinforcement Learning from Human Feedback pipeline: SFT → Reward Model → DPO.
> Config-driven, LoRA-supported, WandB-integrated.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Pipeline Overview

```
Step 1: SFT          Step 2: Reward Model     Step 3: DPO / PPO
─────────────        ────────────────────     ─────────────────
Raw base model  →    Train on preference  →   Align policy with
+ instruction data   pairs (chosen/rejected)  reward signal
```

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Generate a test dataset
python -c "from abark_rlhf.data.dataset_utils import create_dummy_preference_dataset; create_dummy_preference_dataset(200)"

# Run SFT
python scripts/run_sft.py --config configs/sft_config.yaml

# Run DPO
python scripts/run_dpo.py --config configs/dpo_config.yaml

# Train reward model
python scripts/run_reward_model.py --config configs/reward_model_config.yaml
```

---

## Dataset Format

### Preference pairs (for DPO + Reward Model)
```jsonl
{"prompt": "Explain quantum computing", "chosen": "Quantum computing uses qubits...", "rejected": "It's just fast computers."}
```

### SFT data
```jsonl
{"prompt": "Write a haiku about AI", "response": "Silicon dreams hum / Data flows like mountain streams / Minds made from numbers"}
```

---

## Config-Driven Training

All training is controlled via YAML configs in `configs/`:

| Config | Stage | Key params |
|---|---|---|
| `sft_config.yaml` | SFT | `base_model`, `max_seq_length`, `lora_r` |
| `reward_model_config.yaml` | Reward Model | `base_model`, `max_length`, `lora_r` |
| `dpo_config.yaml` | DPO | `base_model`, `beta`, `max_length` |

---

## Architecture

```
abark_rlhf/
├── data/
│   └── dataset_utils.py        # Load JSONL, split, format for HF TRL
├── sft/
│   └── trainer.py              # SFTModelTrainer + SFTConfig
├── reward_model/
│   └── trainer.py              # RewardModelTrainer + RewardModelConfig
├── dpo/
│   └── trainer.py              # DPOTrainer + DPOConfig
└── ppo/
    └── trainer.py              # PPOTrainer (Ray + vLLM, coming soon)
scripts/
├── run_sft.py
├── run_dpo.py
└── run_reward_model.py
configs/
├── sft_config.yaml
├── dpo_config.yaml
└── reward_model_config.yaml
```

---

## Hardware Requirements

| Stage | Minimum GPU | Recommended |
|---|---|---|
| SFT (1B model, LoRA) | 8GB VRAM | A100 40GB |
| Reward Model (1B) | 8GB VRAM | A100 40GB |
| DPO (7B, LoRA) | 2× A100 80GB | 4× A100 |

---

## References & Inspiration

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — Ray + vLLM distributed RLHF
- [RLHFlow/RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling) — Bradley-Terry reward models
- [HuggingFace TRL](https://github.com/huggingface/trl) — PPO, DPO, SFT trainers

---

## License

MIT © [AbArk](https://github.com/AbArk)
