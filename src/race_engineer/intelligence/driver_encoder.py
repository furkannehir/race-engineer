"""Optional offline CPU encoder; PyTorch is imported only in the isolated runtime."""

import json
import os
from pathlib import Path
from typing import Any

from race_engineer.intelligence.driver_actions import (
    LABELS,
    MODEL_FILES,
    MODEL_REPO,
    MODEL_REVISION,
)

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/driver-interpreter-pilot/base"


class Encoder:
    def __init__(self, threads: int = 8, *, base: Path = BASE) -> None:
        if not 1 <= threads <= 64:
            raise ValueError("threads must be between 1 and 64")
        path = base.resolve()
        if not path.is_relative_to(ROOT / "data"):
            raise ValueError("base checkpoint must stay inside repository data")
        if any(not (path / filename).is_file() for filename in (*MODEL_FILES, "provenance.json")):
            raise ValueError("run scripts/setup_driver_interpreter.py first")
        if json.loads((path / "provenance.json").read_text(encoding="utf-8")) != {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
        }:
            raise ValueError("unexpected encoder checkpoint provenance")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

        self.torch: Any = torch
        torch.set_num_threads(threads)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.tokenizer.truncation_side = "left"  # Preserve the current turn and latest dialogue.
        self.backbone: Any = (
            AutoModel.from_pretrained(
                str(path),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .to(device="cpu", dtype=torch.float32)
            .eval()
        )
        if self.backbone.config.model_type != "bert" or self.backbone.config.hidden_size != 384:
            raise ValueError("unexpected encoder architecture")
        self.head: Any = torch.nn.Linear(384, len(LABELS))

    def tokenize(self, texts: list[str]) -> Any:
        return self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )

    def embedding(self, encoded: Any) -> Any:
        states = self.backbone(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
        mean = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.torch.nn.functional.normalize(mean, p=2, dim=1)

    def encode(self, texts: list[str], batch_size: int = 32) -> Any:
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batches.append(self.embedding(self.tokenize(texts[start : start + batch_size])))
        # Ordinary tensors, not inference tensors: a trained head needs autograd on its weights.
        return self.torch.cat(batches).clone()

    def adapt_last_layers(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for layer in self.backbone.encoder.layer[-2:]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        # No dropout differences between cached frozen control and adapted experiment.
        # eval() disables dropout, not gradients. This is a declared pilot choice.
        self.backbone.eval()

    def snapshot(self, *, include_backbone: bool) -> dict[str, Any]:
        result = {
            f"head.{name}": value.detach().cpu().clone()
            for name, value in self.head.state_dict().items()
        }
        if include_backbone:
            trainable = {
                name
                for name, parameter in self.backbone.named_parameters()
                if parameter.requires_grad
            }
            result.update(
                {
                    f"backbone.{name}": value.detach().cpu().clone()
                    for name, value in self.backbone.state_dict().items()
                    if name in trainable
                }
            )
        return result

    def restore(self, weights: dict[str, Any]) -> None:
        allowed = {"head.weight", "head.bias"}
        for index in (10, 11):
            allowed.update(
                f"backbone.encoder.layer.{index}.{name}"
                for name in self.backbone.encoder.layer[index].state_dict()
            )
        if set(weights) not in ({"head.weight", "head.bias"}, allowed):
            raise ValueError("unexpected adapter tensors")
        backbone_weights = {
            name.removeprefix("backbone."): value
            for name, value in weights.items()
            if name.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_weights, strict=False)
        self.head.load_state_dict(
            {
                name.removeprefix("head."): value
                for name, value in weights.items()
                if name.startswith("head.")
            }
        )

    def probabilities(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        values: list[list[float]] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                values.extend(
                    self.torch.sigmoid(
                        self.head(
                            self.embedding(
                                self.tokenize(texts[start : start + batch_size]),
                            )
                        )
                    ).tolist()
                )
        return values
