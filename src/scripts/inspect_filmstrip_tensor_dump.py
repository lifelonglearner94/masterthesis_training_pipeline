"""Inspect filmstrip tensor dumps saved by src.utils.filmstrip_rollout.

Example:
    /home/marcel/code/IU_projects/masterthesis_goldstandard_repo/.venv/bin/python \
        src/scripts/inspect_filmstrip_tensor_dump.py \
        outputs/outputs/2026-03-21/11-06-41_Hybrid/filmstrips/after_base/base_tensors.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


EXPECTED_KEYS = ("gt", "pred")


def _summarize_tensor(name: str, tensor: torch.Tensor) -> list[str]:
    tensor_cpu = tensor.detach().cpu()
    lines = [
        f"{name}:",
        f"  shape={tuple(tensor_cpu.shape)}",
        f"  dtype={tensor_cpu.dtype}",
        f"  device={tensor.device}",
        f"  finite={bool(torch.isfinite(tensor_cpu).all().item())}",
    ]

    if tensor_cpu.numel() > 0:
        tensor_float = tensor_cpu.float()
        lines.extend(
            [
                f"  min={tensor_float.min().item():.6g}",
                f"  max={tensor_float.max().item():.6g}",
                f"  mean={tensor_float.mean().item():.6g}",
                f"  std={tensor_float.std().item():.6g}",
            ]
        )

    return lines


def _validate_payload(payload: Any) -> list[str]:
    issues: list[str] = []

    if not isinstance(payload, dict):
        return [f"payload is {type(payload).__name__}, expected dict with keys {EXPECTED_KEYS}"]

    missing = [key for key in EXPECTED_KEYS if key not in payload]
    if missing:
        issues.append(f"missing required keys: {missing}")

    extra = sorted(key for key in payload if key not in EXPECTED_KEYS)
    if extra:
        issues.append(f"unexpected extra keys: {extra}")

    gt = payload.get("gt")
    pred = payload.get("pred")

    for name, value in (("gt", gt), ("pred", pred)):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            issues.append(f"{name} is {type(value).__name__}, expected torch.Tensor")
            continue
        if value.ndim != 3:
            issues.append(f"{name} has ndim={value.ndim}, expected 3 ([T, N, D])")
        if value.numel() == 0:
            issues.append(f"{name} is empty")
        if not torch.isfinite(value.detach().cpu()).all().item():
            issues.append(f"{name} contains NaN or inf values")

    if isinstance(gt, torch.Tensor) and isinstance(pred, torch.Tensor):
        if gt.shape != pred.shape:
            issues.append(f"shape mismatch: gt={tuple(gt.shape)} vs pred={tuple(pred.shape)}")
        elif gt.ndim == 3:
            t_steps, patch_count, feature_dim = gt.shape
            if t_steps != 8:
                issues.append(f"expected 8 frames for current filmstrip pipeline, found {t_steps}")
            if patch_count != 256:
                issues.append(
                    f"expected 256 patches for 16x16 filmstrip grids, found {patch_count}"
                )
            if feature_dim <= 0:
                issues.append(f"feature dimension must be positive, found {feature_dim}")

    return issues


def inspect_tensor_dump(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)

    print(f"file: {path}")
    print(f"payload_type: {type(payload).__name__}")

    if isinstance(payload, dict):
        print(f"keys: {sorted(payload.keys())}")

    print()

    issues = _validate_payload(payload)

    if isinstance(payload, dict):
        for key in EXPECTED_KEYS:
            value = payload.get(key)
            if isinstance(value, torch.Tensor):
                for line in _summarize_tensor(key, value):
                    print(line)
                print()

    if issues:
        print("result: FAIL")
        print("reason: tensor dump does not match the current filmstrip tensor contract")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("result: PASS")
    print("reason: file contains the raw gt/pred latent tensors expected by the filmstrip renderer")
    print("rerendering: POSSIBLE")
    print(
        "note: reproducing the exact original PNG also requires the fitted PCA basis used during filmstrip generation; this .pt file stores the latent tensors, not the PCA object"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a filmstrip tensor dump and verify it matches the renderer contract."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to a *_tensors.pt file produced by the filmstrip pipeline.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.path.expanduser().resolve()

    if not path.exists():
        print(f"error: file not found: {path}")
        return 2
    if not path.is_file():
        print(f"error: not a file: {path}")
        return 2

    return inspect_tensor_dump(path)


if __name__ == "__main__":
    raise SystemExit(main())
