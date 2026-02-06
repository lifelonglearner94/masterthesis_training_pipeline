"""Quick smoke test for AC-HOPE-ViT architecture."""
import torch

from src.models.hope import ACHOPEModule, ACHOPEViT, ac_hope_vit

print("✅ Import successful")

# Instantiate the core model
model = ac_hope_vit(
    img_size=(256, 256),
    patch_size=16,
    num_timesteps=8,
    embed_dim=1024,
    predictor_embed_dim=384,
    depth=6,  # Use 6 layers for quick test
    num_heads=16,
    action_embed_dim=2,
    use_rope=True,
    is_frame_causal=True,
    titan_hidden_multiplier=4,
    titan_layers=2,
    titan_grad_clip_inner=1.0,
    self_mod_dim=64,
    surprise_threshold=0.0,
    drop_path_rate=0.1,
    log_hope_diagnostics=True,
)
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ ACHOPEViT instantiated: {total_params:,} parameters")

# Test forward pass (inference)
B, T, N, D = 2, 7, 256, 1024
x = torch.randn(B, T * N, D)
actions = torch.randn(B, T, 2)
states = torch.randn(B, T, 2)

with torch.no_grad():
    out = model(x, actions, states)
print(f"✅ Forward pass (inference): input {x.shape} → output {out.shape}")
assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

# Test forward pass (training — needed for DGD inner-loop)
model.train()
model.reset_all_memories()  # Initialize active weights for functional forward
x2 = torch.randn(2, 7 * 256, 1024, requires_grad=False)
actions2 = torch.randn(2, 7, 2)
states2 = torch.randn(2, 7, 2)
out2 = model(x2, actions2, states2)
loss = out2.mean()
loss.backward()
print(f"✅ Forward+backward pass (training): loss={loss.item():.6f}")

# Test config summary (Criticism §1)
config = model.get_config_summary()
print(f"✅ Config summary: {config}")

# Test diagnostics (Criticism §1)
diag = model.get_all_diagnostics()
print(f"✅ Diagnostics: {len(diag)} metrics")
for k, v in diag.items():
    print(f"    {k}: {v:.6f}")

# Test parameter groups
groups = model.get_parameter_groups()
for g in groups:
    n = sum(p.numel() for p in g["params"])
    print(f'  Group "{g["group_name"]}": {n:,} params')

# Test with use_rope=False (Criticism §2 ablation)
model_no_rope = ac_hope_vit(
    img_size=(256, 256),
    patch_size=16,
    num_timesteps=8,
    embed_dim=1024,
    predictor_embed_dim=384,
    depth=2,
    num_heads=16,
    action_embed_dim=2,
    use_rope=False,
    is_frame_causal=True,
)
with torch.no_grad():
    out_nr = model_no_rope(x, actions, states)
print(f"✅ Forward pass (no RoPE): output {out_nr.shape}")

# Test Lightning module instantiation
module = ACHOPEModule(
    img_size=(256, 256),
    patch_size=16,
    num_timesteps=8,
    embed_dim=1024,
    predictor_embed_dim=384,
    depth=2,
    num_heads=16,
    action_embed_dim=2,
    use_rope=True,
    T_teacher=7,
    T_rollout=7,
    context_frames=1,
)
print(f"✅ ACHOPEModule instantiated: {sum(p.numel() for p in module.parameters()):,} params")

print()
print("🎉 All tests passed!")
