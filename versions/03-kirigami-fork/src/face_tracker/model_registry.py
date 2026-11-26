"""Compact HSEmotion variants supported by the pinned ONNX runtime package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    expected_classes: int


MODEL_SPECS = (
    ModelSpec("enet_b0_8_best_vgaf", "B0 · VGAF (current baseline)", 8),
    ModelSpec("enet_b0_8_best_afew", "B0 · AFEW", 8),
    ModelSpec("enet_b0_8_va_mtl", "B0 · VA multi-task", 8),
    ModelSpec("enet_b2_8", "B2 · AffectNet-8", 8),
)


def get_model_spec(name: str) -> ModelSpec:
    for spec in MODEL_SPECS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unsupported expression model: {name}")
