"""Minimal local Transformers surface required by TraceFormer."""

from .configuration_gpt2 import GPT2Config
from . import modeling_gpt2_pair_dpo

__all__ = ["GPT2Config", "modeling_gpt2_pair_dpo"]
