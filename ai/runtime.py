"""Shared helpers to load Hugging Face models safely."""

import logging
from typing import Any

import torch

from ai.settings import ModelsConfig

logger: logging.Logger = logging.getLogger(__name__)


def resolve_device(setting: str) -> str:
    """Return `cpu` or `cuda`. `auto` uses a GPU if one exists."""
    assert setting in ("auto", "cpu", "cuda"), f"unknown device setting: {setting}"
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if setting == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is cuda, but no GPU is available")
    return setting


def configure_threads(cpu_threads: int) -> None:
    """Set the number of CPU threads for torch. 0 keeps the default."""
    assert cpu_threads >= 0, "cpu_threads must not be negative"
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        logger.info("Torch uses %d CPU thread(s)", cpu_threads)


def load_kwargs(config: ModelsConfig, revision: str) -> dict[str, Any]:
    """Return the safe arguments for `from_pretrained`.

    The model is fixed to one revision. Only safetensors files are loaded.
    Code from the model repository is never run.
    """
    assert revision, "revision is required"
    return {
        "revision": revision,
        "local_files_only": config.local_files_only,
        "trust_remote_code": False,
        "use_safetensors": True,
    }


def short_model_name(model_id: str, revision: str) -> str:
    """Return a name such as `dinov2-small@ed25f3a3` for the database."""
    return f"{model_id.split('/')[-1]}@{revision[:8]}"
