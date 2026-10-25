"""Shared model, distributed, checkpoint, and feature utilities."""

from __future__ import annotations

import os
import random
import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist

import transformers
from modelss.point_mamba import PointFORGPT


def project_root() -> Path:
    return Path(__file__).resolve().parent


def resolve_inside_project(value: str | Path) -> Path:
    root = project_root()
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path must remain inside {root}: {path}") from error
    return path


def set_seed(seed: int, rank: int = 0) -> None:
    actual = int(seed) + int(rank)
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    torch.cuda.manual_seed_all(actual)


def init_distributed() -> Tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
    else:
        rank, world_size, local_rank = 0, 1, 0
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required by the morphology encoder")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def _clean_state_dict(state: Dict[str, torch.Tensor]) -> OrderedDict:
    clean = OrderedDict()
    for key, value in state.items():
        clean[key[7:] if key.startswith("module.") else key] = value
    return clean


def load_component(module: torch.nn.Module, path: Path) -> None:
    state = torch.load(str(path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = module.load_state_dict(_clean_state_dict(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible component checkpoint {path}: missing={missing}, unexpected={unexpected}"
        )


def build_from_checkpoint(checkpoint: Path, device: torch.device):
    required = ["config.json", "pytorch_model.bin", "encoder.pth", "img_mlp.pth"]
    absent = [name for name in required if not (checkpoint / name).is_file()]
    if absent:
        raise FileNotFoundError(f"Incomplete checkpoint {checkpoint}; missing {absent}")

    model, loading_info = transformers.modeling_gpt2_pair_dpo.NeuronGPTLMHeadModel.from_pretrained(
        str(checkpoint), output_loading_info=True
    )
    serious = {
        key: value
        for key, value in loading_info.items()
        if key in {"missing_keys", "unexpected_keys", "error_msgs"} and value
    }
    if serious:
        raise RuntimeError(f"Transformer checkpoint mismatch: {serious}")

    encoder = PointFORGPT()
    image_state = torch.load(str(checkpoint / "img_mlp.pth"), map_location="cpu")
    image_state = _clean_state_dict(image_state)
    weight_keys = [key for key in image_state if key.endswith("weight")]
    if len(weight_keys) == 1:
        output_dim, input_dim = image_state[weight_keys[0]].shape
        image_mlp = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim), torch.nn.ReLU())
    elif len(weight_keys) == 2:
        hidden_dim, input_dim = image_state[weight_keys[0]].shape
        output_dim, second_input = image_state[weight_keys[1]].shape
        if second_input != hidden_dim:
            raise RuntimeError(f"Unsupported image MLP checkpoint: {checkpoint}")
        image_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )
    else:
        raise RuntimeError(f"Unsupported image MLP checkpoint: {checkpoint}")
    load_component(encoder, checkpoint / "encoder.pth")
    missing, unexpected = image_mlp.load_state_dict(image_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible image MLP checkpoint: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device), encoder.to(device), image_mlp.to(device)


def build_from_scratch(config_path: Path, device: torch.device):
    """Randomly initialize TraceFormer and its multimodal feature modules."""
    config = transformers.GPT2Config.from_json_file(str(config_path))
    model = transformers.modeling_gpt2_pair_dpo.NeuronGPTLMHeadModel(config)
    encoder = PointFORGPT()
    image_mlp = torch.nn.Sequential(torch.nn.Linear(256, 1024), torch.nn.ReLU())
    return model.to(device), encoder.to(device), image_mlp.to(device)


def clone_modules(model, encoder, image_mlp):
    """Create an identical reference branch before any optimization step."""
    return copy.deepcopy(model), copy.deepcopy(encoder), copy.deepcopy(image_mlp)


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def encode_state(batch, encoder, image_mlp, device, add_image_feature: bool):
    history_points = batch["history_points"].to(device)
    candidate_points = batch["candidate_points"].to(device)
    history_features = encoder(history_points)
    candidate_features = encoder(candidate_points)
    if isinstance(history_features, tuple):
        history_features = history_features[0]
        candidate_features = candidate_features[0]

    inputs = torch.cat([history_features, candidate_features], dim=0).unsqueeze(0)
    if add_image_feature:
        history_images = image_mlp(batch["history_images"].to(device))
        candidate_images = image_mlp(batch["candidate_images"].to(device))
        image_features = torch.cat([history_images, candidate_images], dim=0).unsqueeze(0)
        inputs = inputs + image_features

    return (
        inputs,
        batch["labels"].to(device),
        batch["coordinates"].to(device).unsqueeze(0),
        torch.tensor([batch["history_length"]], dtype=torch.float32, device=device),
        torch.tensor([batch["gt_index"]], dtype=torch.long, device=device),
    )


def forward_scores(model, encoded):
    inputs, labels, coordinates, history_length, gt_index = encoded
    loss, predictions, scores = model(
        input_embeddings=inputs,
        batch_img_embeds=None,
        labels=labels,
        corrs=coordinates,
        tracing_status=history_length,
        gt_id=gt_index,
    )
    return loss, predictions[:, 0], scores.reshape(-1)


def save_checkpoint(output: Path, model, encoder, image_mlp) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unwrap(model).save_pretrained(str(output))
    torch.save(unwrap(encoder).state_dict(), output / "encoder.pth")
    torch.save(unwrap(image_mlp).state_dict(), output / "img_mlp.pth")


def all_reduce_sum(values: Iterable[float], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(list(values), dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor
