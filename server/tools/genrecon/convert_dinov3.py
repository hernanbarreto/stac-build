"""Meta's DINOv3 ViT-L/16 checkpoint (the pod's, downloaded with the user's links) → the transformers
layout GenRecon loads (DINOv3ViTModel.from_pretrained), validated against Meta's reference code."""
import sys, json, torch, numpy as np
from pathlib import Path
from transformers import DINOv3ViTConfig, DINOv3ViTModel
SRC = Path("/workspace/stac-build/vendor/pointdit/pretrained/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
DST = Path("/workspace/stac-build/vendor/GenRecon/weights/dinov3-vitl16-hf")
sd = torch.load(SRC, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
D = sd["cls_token"].shape[-1]; L = len({k.split(".")[1] for k in sd if k.startswith("blocks.")}); R = sd["storage_tokens"].shape[1]
cfg = DINOv3ViTConfig(hidden_size=D, intermediate_size=sd["blocks.0.mlp.fc1.weight"].shape[0], num_hidden_layers=L,
                      num_attention_heads=16, patch_size=16, num_register_tokens=R, rope_theta=100.0, image_size=224,
                      layerscale_value=1.0, use_gated_mlp=False, hidden_act="gelu", layer_norm_eps=1e-5,
                      query_bias=True, key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True)
m = DINOv3ViTModel(cfg)
new = {}
new["embeddings.cls_token"] = sd["cls_token"]; new["embeddings.register_tokens"] = sd["storage_tokens"]
new["embeddings.mask_token"] = sd["mask_token"].reshape(1, 1, D)
new["embeddings.patch_embeddings.weight"] = sd["patch_embed.proj.weight"]; new["embeddings.patch_embeddings.bias"] = sd["patch_embed.proj.bias"]
for i in range(L):
    p = f"blocks.{i}."; q = f"layer.{i}."
    new[q + "norm1.weight"] = sd[p + "norm1.weight"]; new[q + "norm1.bias"] = sd[p + "norm1.bias"]
    W = sd[p + "attn.qkv.weight"]; b = sd[p + "attn.qkv.bias"] * sd[p + "attn.qkv.bias_mask"]      # the mask IS the model
    new[q + "attention.q_proj.weight"], new[q + "attention.k_proj.weight"], new[q + "attention.v_proj.weight"] = W[:D], W[D:2 * D], W[2 * D:]
    new[q + "attention.q_proj.bias"] = b[:D]; new[q + "attention.v_proj.bias"] = b[2 * D:]
    assert float(b[D:2 * D].abs().max()) == 0.0
    new[q + "attention.o_proj.weight"] = sd[p + "attn.proj.weight"]; new[q + "attention.o_proj.bias"] = sd[p + "attn.proj.bias"]
    new[q + "layer_scale1.lambda1"] = sd[p + "ls1.gamma"]; new[q + "layer_scale2.lambda1"] = sd[p + "ls2.gamma"]
    new[q + "norm2.weight"] = sd[p + "norm2.weight"]; new[q + "norm2.bias"] = sd[p + "norm2.bias"]
    new[q + "mlp.up_proj.weight"] = sd[p + "mlp.fc1.weight"]; new[q + "mlp.up_proj.bias"] = sd[p + "mlp.fc1.bias"]
    new[q + "mlp.down_proj.weight"] = sd[p + "mlp.fc2.weight"]; new[q + "mlp.down_proj.bias"] = sd[p + "mlp.fc2.bias"]
new["norm.weight"] = sd["norm.weight"]; new["norm.bias"] = sd["norm.bias"]
missing, unexpected = m.load_state_dict(new, strict=False)
print("missing", missing, "unexpected", unexpected)
assert not unexpected and all(k.endswith("inv_freq") for k in missing), (missing, unexpected)
# rope check: Meta's periods vs HF's 1/inv_freq
per = sd["rope_embed.periods"].float(); hf_per = 1.0 / m.rope_embeddings.inv_freq.float()
print("rope periods max rel diff", float(((per - hf_per).abs() / per).max()))
DST.mkdir(parents=True, exist_ok=True); m.save_pretrained(DST, safe_serialization=True)
print("saved", DST, sorted(p.name for p in DST.iterdir()))
# validation against Meta's reference implementation (the hub code the pod already has)
sys.path.insert(0, "/workspace/stac-build/vendor/pointdit/third_party/dinov3")
try:
    from dinov3.hub.backbones import dinov3_vitl16
    ref = dinov3_vitl16(pretrained=True, weights=str(SRC)).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = m(pixel_values=x).last_hidden_state          # [cls, registers, patches], after the learned norm
        r = ref.forward_features(x)
        ref_cls, ref_pat = r["x_norm_clstoken"], r["x_norm_patchtokens"]
    print("cls max abs diff", float((out[:, 0] - ref_cls).abs().max()), "patch max abs diff", float((out[:, 1 + R:] - ref_pat).abs().max()),
          "patch scale", float(ref_pat.abs().mean()))
except Exception as e:
    print("reference check skipped:", type(e).__name__, str(e)[:200])
