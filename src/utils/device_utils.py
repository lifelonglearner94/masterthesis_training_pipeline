"""Device utilities for multi-platform support (CUDA > MPS > CPU).

This module provides utilities for detecting and configuring the best available
hardware accelerator, with automatic fallbacks for different platforms.
"""

import logging
import warnings
from typing import Final, Literal

import torch

log = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]
Precision = Literal["16-mixed", "32"]
Accelerator = Literal["gpu", "mps", "cpu"]

# Constants
MIN_COMPUTE_CAPABILITY: Final = 7
LOG_SEPARATOR_LENGTH: Final = 60
FLASH_ATTENTION_COMPUTE_CAPABILITY: Final = 7.5


def get_best_device() -> DeviceType:
    """Detect the best available device with priority: CUDA > MPS > CPU.

    Returns:
        The device type string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_accelerator() -> Accelerator:
    """Get the PyTorch Lightning accelerator string for the best available device.

    Returns:
        Accelerator string compatible with PyTorch Lightning Trainer
    """
    device = get_best_device()
    if device == "cuda":
        return "gpu"
    if device == "mps":
        return "mps"
    return "cpu"


def get_precision(device: DeviceType | None = None) -> Precision:
    """Get the recommended precision setting for the given device.

    Args:
        device: Device type. If None, auto-detects the best device.

    Returns:
        Precision string compatible with PyTorch Lightning Trainer
    """
    if device is None:
        device = get_best_device()

    if device == "cuda":
        return "16-mixed"  # Mixed precision works well on CUDA
    return "32"  # MPS has limited fp16 support; CPU doesn't benefit


def check_flash_attention_support() -> bool:
    """Check if FlashAttention is available (CUDA-only feature).

    Returns:
        True if FlashAttention is available, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    try:
        if torch.cuda.get_device_capability()[0] >= MIN_COMPUTE_CAPABILITY:
            return True
    except Exception:
        return False

    return False


def _log_flash_attention_warning(device: str) -> None:
    """Log a warning about FlashAttention unavailability for non-CUDA devices."""
    warnings.warn(
        f"FlashAttention is not available on {device}. "
        "Using PyTorch's scaled_dot_product_attention with automatic backend selection.",
        UserWarning,
        stacklevel=3,
    )


def log_device_info() -> DeviceType:
    """Log information about the detected hardware and return the device type.

    This function prints detailed hardware information at startup, including
    warnings for non-CUDA systems about potential limitations.

    Returns:
        The detected device type
    """
    device = get_best_device()
    accelerator = get_accelerator()
    precision = get_precision(device)

    separator = "=" * LOG_SEPARATOR_LENGTH
    log.info(separator)
    log.info("Hardware Detection")
    log.info(separator)

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        cuda_version = torch.version.cuda
        log.info(f"Device: CUDA GPU")
        log.info(f"GPU: {gpu_name}")
        log.info(f"GPU Count: {gpu_count}")
        log.info(f"CUDA Version: {cuda_version}")
        log.info(f"Precision: {precision} (mixed precision enabled)")

        if check_flash_attention_support():
            log.info("FlashAttention: Available")
        else:
            log.info(f"FlashAttention: Not available (compute capability < {FLASH_ATTENTION_COMPUTE_CAPABILITY})")

    elif device == "mps":
        log.info("Device: Apple Silicon (MPS)")
        log.info(f"Precision: {precision}")
        log.warning(
            "Running on Apple MPS. Some features may have reduced performance. "
            "Mixed precision is disabled due to limited fp16 support."
        )
        _log_flash_attention_warning("MPS")

    else:
        log.info("Device: CPU")
        log.info(f"Precision: {precision}")
        log.warning(
            "Running on CPU. Training will be significantly slower than GPU. "
            "Consider using a CUDA GPU or Apple Silicon Mac for better performance."
        )
        _log_flash_attention_warning("CPU")

    log.info(f"PyTorch Lightning Accelerator: {accelerator}")
    log.info(separator)

    return device


def should_pin_memory() -> bool:
    """Determine if pin_memory should be enabled for DataLoaders.

    pin_memory only provides benefits when transferring data to CUDA GPUs.

    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()
