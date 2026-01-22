from typing import Any, Optional
import time
import logging
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

try:
    import psutil
except ImportError:
    psutil = None

log = logging.getLogger(__name__)


class TensorStats:
    """Helper to compute and format tensor statistics."""

    @staticmethod
    def stats(t: torch.Tensor, name: str = "tensor") -> str:
        if t is None:
            return f"{name}: None"
        if not isinstance(t, torch.Tensor):
            return f"{name}: {type(t)}"
        if t.numel() == 0:
            return f"{name}: empty tensor"

        with torch.no_grad():
            t_float = t.float()
            return (
                f"{name}: shape={list(t.shape)}, dtype={t.dtype}, "
                f"min={t_float.min().item():.6f}, max={t_float.max().item():.6f}, "
                f"mean={t_float.mean().item():.6f}, std={t_float.std().item():.6f}, "
                f"norm={t_float.norm().item():.6f}"
            )


class ForwardHookLogger:
    """Logs forward pass details for each layer."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def __call__(self, module: nn.Module, input_tuple: tuple, output: Any) -> None:
        log.info(f"  [FWD] {self.module_name}")

        # Log input shapes and stats
        for i, inp in enumerate(input_tuple):
            if isinstance(inp, torch.Tensor):
                log.info(f"    Input[{i}]: {TensorStats.stats(inp, 'x')}")

        # Log output shapes and stats
        if isinstance(output, torch.Tensor):
            log.info(f"    Output: {TensorStats.stats(output, 'y')}")
        elif isinstance(output, tuple):
            for i, out in enumerate(output):
                if isinstance(out, torch.Tensor):
                    log.info(f"    Output[{i}]: {TensorStats.stats(out, 'y')}")


class BackwardHookLogger:
    """Logs backward pass gradient details for each layer."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def __call__(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        log.info(f"  [BWD] {self.module_name}")

        # Log gradient outputs (dL/dy)
        for i, grad in enumerate(grad_output):
            if grad is not None and isinstance(grad, torch.Tensor):
                log.info(f"    grad_output[{i}]: {TensorStats.stats(grad, 'dL/dy')}")

        # Log gradient inputs (dL/dx)
        for i, grad in enumerate(grad_input):
            if grad is not None and isinstance(grad, torch.Tensor):
                log.info(f"    grad_input[{i}]: {TensorStats.stats(grad, 'dL/dx')}")


class VerboseLoggingCallback(Callback):
    """Callback to log EXTREMELY detailed information about training."""

    def __init__(
        self,
        log_memory: bool = True,
        log_data: bool = True,
        log_forward_pass: bool = True,
        log_backward_pass: bool = True,
        log_gradients: bool = True,
        log_weights: bool = True,
        log_every_n_batches: int = 1,  # Log every N batches (1 = every batch)
        max_layers_to_log: int = 50,  # Limit number of layers to avoid extreme spam
    ):
        super().__init__()
        self.log_memory = log_memory
        self.log_data = log_data
        self.log_forward_pass = log_forward_pass
        self.log_backward_pass = log_backward_pass
        self.log_gradients = log_gradients
        self.log_weights = log_weights
        self.log_every_n_batches = log_every_n_batches
        self.max_layers_to_log = max_layers_to_log
        self.forward_hooks = []
        self.backward_hooks = []
        self.epoch_start_time = time.time()

    def _is_any_logging_enabled(self) -> bool:
        """Check if any logging option is enabled."""
        return any([
            self.log_memory,
            self.log_data,
            self.log_forward_pass,
            self.log_backward_pass,
            self.log_gradients,
            self.log_weights,
        ])

    def _should_log_batch(self, batch_idx: int) -> bool:
        return self._is_any_logging_enabled() and batch_idx % self.log_every_n_batches == 0

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward and backward hooks on model layers."""
        if not self.log_forward_pass and not self.log_backward_pass:
            return

        layer_count = 0

        for name, module in model.named_modules():
            if layer_count >= self.max_layers_to_log:
                break

            # Only hook "interesting" layers
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
                if self.log_forward_pass:
                    handle = module.register_forward_hook(ForwardHookLogger(name))
                    self.forward_hooks.append(handle)

                if self.log_backward_pass:
                    handle = module.register_full_backward_hook(BackwardHookLogger(name))
                    self.backward_hooks.append(handle)

                layer_count += 1

        if layer_count > 0:
            log.info(f"Registered hooks on {layer_count} layers")

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.forward_hooks:
            handle.remove()
        for handle in self.backward_hooks:
            handle.remove()
        self.forward_hooks = []
        self.backward_hooks = []

    @rank_zero_only
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._is_any_logging_enabled():
            return

        log.info("=" * 100)
        log.info("TRAINING STARTED - ULTRA VERBOSE MODE")
        log.info("=" * 100)
        log.info(f"Model: {pl_module.__class__.__name__}")

        # Detailed parameter count
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        log.info(f"Total parameters: {total_params:,} ({total_params * 4 / 1024**3:.2f} GB in fp32)")
        log.info(f"Trainable parameters: {trainable_params:,}")

        # Log model architecture
        log.info("\n--- MODEL ARCHITECTURE ---")
        for name, module in pl_module.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    log.info(f"  {name}: {module.__class__.__name__} ({param_count:,} params)")

        # Log precision and device
        log.info(f"\n--- HARDWARE CONFIG ---")
        log.info(f"Precision: {trainer.precision}")
        log.info(f"Accelerator: {trainer.accelerator}")
        log.info(f"Strategy: {trainer.strategy}")
        log.info(f"Num devices: {trainer.num_devices}")

        # Log optimizer config
        if trainer.optimizers:
            for i, opt in enumerate(trainer.optimizers):
                log.info(f"\n--- OPTIMIZER {i} ---")
                log.info(f"Type: {opt.__class__.__name__}")
                for key, value in opt.defaults.items():
                    log.info(f"  {key}: {value}")

        # Register hooks for detailed layer logging
        if self.log_forward_pass or self.log_backward_pass:
            self._register_hooks(pl_module)

        log.info("=" * 100)

    @rank_zero_only
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._remove_hooks()
        if not self._is_any_logging_enabled():
            return

        log.info("=" * 100)
        log.info("TRAINING ENDED")
        log.info("=" * 100)

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start_time = time.time()
        if not self._is_any_logging_enabled():
            return

        log.info(f"\n{'='*50} EPOCH {trainer.current_epoch} {'='*50}")
        if self.log_memory:
            self._log_memory_usage("Epoch Start")

    @rank_zero_only
    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: Any, batch_idx: int) -> None:
        if not self._should_log_batch(batch_idx):
            return

        if self.log_data and isinstance(batch, dict):
            log.info("--- INPUT DATA ---")
            clip_names = batch.get("clip_names", [])
            if clip_names:
                log.info(f"Clips: {clip_names}")

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    log.info(f"  {key}: {TensorStats.stats(value, key)}")

        if self.log_memory:
            self._log_memory_usage("Before Forward")

    @rank_zero_only
    def on_before_backward(self, trainer: L.Trainer, pl_module: L.LightningModule, loss: torch.Tensor) -> None:
        batch_idx = trainer.global_step
        if not self._should_log_batch(batch_idx):
            return

        if self.log_gradients or self.log_backward_pass:
            log.info(f"\n--- LOSS COMPUTATION (Batch {batch_idx}) ---")
            log.info(f"  Loss value: {loss.item():.8f}")
            log.info(f"  Loss tensor: {TensorStats.stats(loss, 'loss')}")
            log.info(f"  Loss requires_grad: {loss.requires_grad}")
            log.info(f"  Loss grad_fn: {loss.grad_fn}")

        if self.log_memory:
            self._log_memory_usage("Before Backward")

    @rank_zero_only
    def on_after_backward(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        batch_idx = trainer.global_step
        if not self._should_log_batch(batch_idx):
            return

        if self.log_gradients:
            log.info("--- GRADIENT STATISTICS ---")
            total_grad_norm = 0.0
            grad_count = 0

            for name, param in pl_module.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm ** 2
                    grad_count += 1

                    # Log individual gradient stats (sample every 10th param to reduce spam)
                    if grad_count <= 20 or grad_count % 50 == 0:
                        log.info(f"  {name}: {TensorStats.stats(param.grad, 'grad')}")

            total_grad_norm = total_grad_norm ** 0.5
            log.info(f"  TOTAL gradient norm: {total_grad_norm:.6f} (across {grad_count} params)")

        if self.log_memory:
            self._log_memory_usage("After Backward")

    @rank_zero_only
    def on_before_optimizer_step(self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer) -> None:
        batch_idx = trainer.global_step
        if not self._should_log_batch(batch_idx):
            return

        if self.log_weights:
            log.info(f"\n--- BEFORE OPTIMIZER STEP (Batch {batch_idx}) ---")
            log.info(f"  Learning rate: {optimizer.param_groups[0]['lr']:.8e}")
            log.info("--- WEIGHT STATISTICS (before update) ---")
            param_count = 0
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    param_count += 1
                    if param_count <= 10 or param_count % 100 == 0:
                        log.info(f"  {name}: {TensorStats.stats(param.data, 'weight')}")

    @rank_zero_only
    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        if not self._should_log_batch(batch_idx):
            return

        if self.log_weights:
            log.info("--- WEIGHT STATISTICS (after update) ---")
            param_count = 0
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    param_count += 1
                    if param_count <= 5:  # Just a sample
                        log.info(f"  {name}: {TensorStats.stats(param.data, 'weight')}")

        if self.log_memory:
            self._log_memory_usage("Batch End")

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._is_any_logging_enabled():
            return

        epoch_time = time.time() - self.epoch_start_time
        log.info(f"\n{'='*50} EPOCH {trainer.current_epoch} END {'='*50}")
        log.info(f"Epoch duration: {epoch_time:.2f} seconds")

        if self.log_memory:
            self._log_memory_usage("Epoch End")

        # Log final weight statistics
        if self.log_weights:
            log.info("\n--- EPOCH END WEIGHT SUMMARY ---")
            for name, param in pl_module.named_parameters():
                if param.requires_grad:
                    log.info(f"  {name}: {TensorStats.stats(param.data, 'weight')}")

    def _log_memory_usage(self, prefix: str):
        msg = f"[MEMORY] {prefix} | "

        # System Memory
        if psutil:
            mem = psutil.virtual_memory()
            msg += f"Sys RAM: {mem.percent}% ({mem.used / 1024**3:.2f}GB used) | "
        else:
            # Fallback for Linux
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                msg += f"Proc Mem: {usage / 1024 / 1024:.2f}GB | "

                with open('/proc/meminfo', 'r') as f:
                    meminfo = {line.split(':')[0]: int(line.split(':')[1].strip().split()[0]) for line in f}
                if 'MemTotal' in meminfo and 'MemAvailable' in meminfo:
                    total = meminfo['MemTotal']
                    avail = meminfo['MemAvailable']
                    used = total - avail
                    percent = (used / total) * 100
                    msg += f"Sys RAM: {percent:.1f}% ({used / 1024 / 1024:.2f}GB used) | "
            except Exception:
                msg += "Mem: N/A | "

        # GPU Memory
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            msg += f"GPU: {allocated:.2f}GB alloc, {reserved:.2f}GB res, {max_allocated:.2f}GB peak"
        elif torch.backends.mps.is_available():
            msg += "MPS Device"
        else:
            msg += "CPU only"

        log.info(msg)
