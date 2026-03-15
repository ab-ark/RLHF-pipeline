"""
Tests for AbArk RLHF Pipeline (CPU-safe, no model downloads).
Run with: pytest tests/ -v
"""

import json
import os
import pytest
import tempfile

from abark_rlhf.data.dataset_utils import (
    PreferenceSample, SFTSample,
    load_preference_jsonl, load_sft_jsonl,
    train_val_split, format_preference_for_hf,
    format_sft_for_hf, create_dummy_preference_dataset,
)
from abark_rlhf.reward_model.trainer import RewardModelConfig
from abark_rlhf.dpo.trainer import DPOConfig
from abark_rlhf.sft.trainer import SFTConfig


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def preference_jsonl(tmp_path):
    data = [
        {"prompt": f"Q{i}?", "chosen": f"Good answer {i}", "rejected": f"Bad answer {i}"}
        for i in range(20)
    ]
    path = tmp_path / "prefs.jsonl"
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return str(path)


@pytest.fixture
def sft_jsonl(tmp_path):
    data = [
        {"prompt": f"Tell me about topic {i}", "response": f"Topic {i} is interesting because..."}
        for i in range(15)
    ]
    path = tmp_path / "sft.jsonl"
    with open(path, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return str(path)


# ── Data Tests ─────────────────────────────────────────────────────────────────

class TestPreferenceSample:
    def test_creation(self):
        s = PreferenceSample(prompt="Q?", chosen="Good", rejected="Bad")
        assert s.prompt == "Q?"
        assert s.metadata == {}

    def test_metadata_default(self):
        s = PreferenceSample(prompt="Q", chosen="A", rejected="B")
        assert isinstance(s.metadata, dict)


class TestLoadPreferenceJsonl:
    def test_loads_correctly(self, preference_jsonl):
        samples = load_preference_jsonl(preference_jsonl)
        assert len(samples) == 20
        assert samples[0].prompt == "Q0?"
        assert samples[0].chosen == "Good answer 0"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_preference_jsonl("nonexistent.jsonl")

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        with open(path, "w") as f:
            f.write('{"prompt": "Q", "chosen": "A", "rejected": "B"}\n')
            f.write('INVALID JSON\n')
            f.write('{"prompt": "Q2", "chosen": "A2", "rejected": "B2"}\n')
        samples = load_preference_jsonl(str(path))
        assert len(samples) == 2


class TestLoadSFTJsonl:
    def test_loads_correctly(self, sft_jsonl):
        samples = load_sft_jsonl(sft_jsonl)
        assert len(samples) == 15
        assert "topic" in samples[0].prompt.lower()


class TestTrainValSplit:
    def test_split_ratio(self, preference_jsonl):
        samples = load_preference_jsonl(preference_jsonl)
        train, val = train_val_split(samples, val_ratio=0.2)
        assert len(train) == 16
        assert len(val) == 4

    def test_split_total(self, preference_jsonl):
        samples = load_preference_jsonl(preference_jsonl)
        train, val = train_val_split(samples, val_ratio=0.1)
        assert len(train) + len(val) == 20

    def test_reproducible_seed(self, preference_jsonl):
        samples = load_preference_jsonl(preference_jsonl)
        t1, v1 = train_val_split(samples, seed=42)
        t2, v2 = train_val_split(samples, seed=42)
        assert [s.prompt for s in t1] == [s.prompt for s in t2]


class TestFormatters:
    def test_preference_format(self):
        samples = [PreferenceSample(prompt="Q?", chosen="Good", rejected="Bad")]
        formatted = format_preference_for_hf(samples)
        assert formatted[0]["prompt"] == "Q?"
        assert formatted[0]["chosen"] == "Good"
        assert "rejected" in formatted[0]

    def test_sft_format_chat_template(self):
        samples = [SFTSample(prompt="Q?", response="A.")]
        formatted = format_sft_for_hf(samples)
        msgs = formatted[0]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "A."


class TestCreateDummyDataset:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "dummy.jsonl")
        result_path = create_dummy_preference_dataset(n=10, path=path)
        assert os.path.exists(result_path)
        samples = load_preference_jsonl(result_path)
        assert len(samples) == 10


# ── Config Tests ───────────────────────────────────────────────────────────────

class TestConfigs:
    def test_reward_model_config_defaults(self):
        cfg = RewardModelConfig()
        assert cfg.use_lora is True
        assert cfg.lora_r == 16
        assert 0 < cfg.val_ratio < 1

    def test_dpo_config_beta(self):
        cfg = DPOConfig(beta=0.2)
        assert cfg.beta == 0.2

    def test_sft_config_defaults(self):
        cfg = SFTConfig()
        assert cfg.packing is True
        assert cfg.max_seq_length == 512
