"""Classification wrapper for benchmarking existing architectures on CL datasets.

Wraps the repo's backbone architectures (AC-ViT, HOPE, Titans, GatedDeltaNet,
RetNet, Transformer++) for image-classification CL benchmarks (Split CIFAR-100,
Permuted MNIST).

Architecture remains identical — only the I/O interface changes:
    1. A ``PatchEmbedding`` layer replaces pre-computed V-JEPA2 features.
    2. The backbone processes the patch tokens (unchanged).
    3. A linear classification head replaces feature prediction.

Actions/states are set to zero — the backbone's action-conditioning path
becomes inert, isolating the core sequence-modelling mechanism.

Usage (via Hydra):
    model:
      _target_: src.models.benchmark_classifier.BackboneClassifierModule
      backbone:
        _target_: src.models.titans.TitansBackbone
        input_dim: 384
        ...
"""

import logging
from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Accuracy

logger = logging.getLogger(__name__)


# =============================================================================
# Patch Embedding
# =============================================================================


class PatchEmbedding(nn.Module):
    """Convert images to a sequence of patch-token embeddings.

    Uses a strided convolution (equivalent to non-overlapping patch extraction
    + linear projection).

    Optionally pads input images to ``pad_to`` before patching (e.g., MNIST
    28×28 → 32×32).

    Args:
        in_channels: Number of input channels (3 for CIFAR, 1 for MNIST).
        embed_dim: Output embedding dimension per patch.
        patch_size: Side length of each square patch.
        img_size: Expected (padded) image side length. Used for sanity checks.
        pad_to: If not None, zero-pad input images to this spatial size.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 384,
        patch_size: int = 4,
        img_size: int = 32,
        pad_to: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.pad_to = pad_to
        self.num_patches = (img_size // patch_size) ** 2  # N
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Patchify images.

        Args:
            x: Images [B, C, H, W].

        Returns:
            Patch tokens [B, N, D] where N = (img_size // patch_size)².
        """
        if self.pad_to is not None:
            _, _, H, W = x.shape
            pad_h = self.pad_to - H
            pad_w = self.pad_to - W
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.proj(x)          # [B, D, H', W']
        x = x.flatten(2)          # [B, D, N]
        x = x.transpose(1, 2)     # [B, N, D]
        return x


# =============================================================================
# Backbone Classification Wrapper  (LightningModule)
# =============================================================================


class BackboneClassifierModule(L.LightningModule):
    """Wraps any backbone architecture for image-classification benchmarks.

    The backbone is passed in fully constructed (Hydra recursive instantiation).
    A :class:`PatchEmbedding` converts images to the token sequence the backbone
    expects (``[B, N, D]``).  Dummy zero-valued actions and states are injected
    so the backbone's action-conditioning path is inert.

    Args:
        backbone: Pre-instantiated backbone (one of the repo's architectures).
        in_channels: Input image channels (3 = RGB, 1 = grayscale).
        img_size: Spatial side length of (padded) images.
        patch_size: Patch side length for the embedding convolution.
        embed_dim: Patch embedding output dimension — must match the backbone's
            input/output dimension (``input_dim`` for conv-encode-decode models,
            ``embed_dim`` for ViT/HOPE models).
        num_classes: Number of classification targets.
        action_dim: Dummy action dimension passed to the backbone.
        pad_to: If set, pad inputs to this size before patch embedding.
            Useful for MNIST (28 → 32).
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        betas: AdamW beta coefficients.
        warmup_epochs: Linear warmup epochs.
        max_epochs: Total training epochs (for cosine schedule).
    """

    def __init__(
        self,
        backbone: nn.Module | None = None,
        in_channels: int = 3,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 384,
        num_classes: int = 100,
        action_dim: int = 1,
        pad_to: int | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        betas: tuple[float, float] | list[float] = (0.9, 0.999),
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])

        # Patch embedding: image → tokens [B, N, D]
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            pad_to=pad_to,
        )

        # The actual architecture under test (None = identity / linear-probe baseline)
        self.backbone = backbone

        # Classification head: pool → logits
        self.head = nn.Linear(embed_dim, num_classes)

        # Config
        self.action_dim = action_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = tuple(betas) if isinstance(betas, list) else betas
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #

    def forward(self, images: Tensor) -> Tensor:
        """Full forward: image → backbone → classification logits.

        Args:
            images: [B, C, H, W] input images.

        Returns:
            Logits [B, num_classes].
        """
        B = images.size(0)
        device = images.device

        # 1. Patchify: [B, C, H, W] → [B, N, D]
        tokens = self.patch_embed(images)

        # 2. Create dummy action / state tensors (T=1 timestep)
        actions = torch.zeros(B, 1, self.action_dim, device=device)
        states = torch.zeros(B, 1, self.action_dim, device=device)

        # 3. Forward through backbone (or identity pass-through)
        if self.backbone is not None:
            # Reset stateful memories if applicable (HOPE memories, etc.)
            if hasattr(self.backbone, "reset_all_memories"):
                self.backbone.reset_all_memories()
            if hasattr(self.backbone, "reset_memory"):
                self.backbone.reset_memory()

            z_out = self.backbone(tokens, actions, states)
        else:
            # No backbone — identity / linear-probe baseline
            z_out = tokens

        # 4. Mean-pool patch tokens → [B, D]
        pooled = z_out.mean(dim=1)

        # 5. Classify
        return self.head(pooled)

    # --------------------------------------------------------------------- #
    # Training / Validation / Test steps
    # --------------------------------------------------------------------- #

    def _shared_step(
        self, batch: tuple[Tensor, Tensor], stage: str,
    ) -> Tensor:
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=-1)

        acc_metric = getattr(self, f"{stage}_acc")
        acc_metric(preds, labels)
        self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, batch_size=images.size(0))
        self.log(f"{stage}/acc", acc_metric, prog_bar=True, sync_dist=True, batch_size=images.size(0))
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    # --------------------------------------------------------------------- #
    # Optimizer
    # --------------------------------------------------------------------- #

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas,
        )
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs - self.warmup_epochs),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
