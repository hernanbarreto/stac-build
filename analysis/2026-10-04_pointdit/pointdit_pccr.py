"""PointDiT-H (DINOv3 ViT-H+/16) on pccr keyframes vs Omega's depth of the same frame.
Writes a side-by-side PNG per frame (RGB | Omega depth | PointDiT depth affine-aligned to Omega | |rel diff|)
and prints alignment residuals, timing and VRAM. Nothing is written into the session."""
import sys, time, types, os, json
import numpy as np, torch
from PIL import Image
ROOT = "/workspace/stac-build/vendor/pointdit"
os.environ.setdefault("DINOV3_REPO", f"{ROOT}/third_party/dinov3"); os.environ.setdefault("DINOV3_WEIGHTS_DIR", f"{ROOT}/pretrained/dinov3")
sys.path.insert(0, ROOT)
from denoiser import Denoiser
S = "/workspace/stac-build/server/projects/pccr/scans/2026-08-31/src_default"
OUT = "/workspace/stac-build/analysis/2026-10-04_pointdit"; os.makedirs(OUT, exist_ok=True)
frames = [int(x) for x in sys.argv[1:]] or [660]
steps = int(os.environ.get("PD_STEPS", "2"))
args = types.SimpleNamespace(model="PointDiT-H/16", img_size=512, attn_dropout=0.0, proj_dropout=0.0, attention_type="torch",
    feature_embedding_type="dinov3_vith16plus", dinov3_use_intermediate_layers=True, dinov3_num_intermediate_layers=4,
    feature_embedding_lr_scale=0.0, P_mean=-0.8, P_std=0.8, t_eps=5e-2, noise_scale=1.0, ema_decay1=0.9999, ema_decay2=0.9999,
    num_sampling_steps=steps, generate_noise_scale=0.0, sample_t_eps=0.0)
t0 = time.time(); m = Denoiser(args)
ck = torch.load(f"{ROOT}/pretrained/pointdith-512-mixdata-nodinov3-cb01dd3b.pth", map_location="cpu", weights_only=False)
state = dict(ck["model"]); ema = ck.get("model_ema1", {}); n_ema = 0
for k, v in ema.items():
    if k in state: state[k] = v; n_ema += 1
res = m.load_state_dict(state, strict=False)
# main.py:704-712 — PointDiT.__init__ re-initialises every module, the frozen encoder included; the DINOv3
# weights are loaded AFTER construction, strictly
dv = torch.load(f"{ROOT}/pretrained/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth", map_location="cpu")
m.net.y_embedder.load_state_dict(dv, strict=True); print(f"DINOv3 H+ loaded: {len(dv)} tensors", flush=True)
bad = [k for k in res.missing_keys if "y_embedder" not in k]
print(f"checkpoint: {len(state)} tensors, {n_ema} from EMA; missing non-encoder {bad[:5]}; unexpected {res.unexpected_keys[:5]}; built {time.time()-t0:.0f}s", flush=True)
m = m.cuda().eval()
def colour(d, lo, hi):
    import matplotlib; matplotlib.use("Agg"); from matplotlib import cm
    x = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1); x[~np.isfinite(d)] = 0
    return (cm.get_cmap("turbo")(x)[..., :3] * 255).astype(np.uint8)
for f in frames:
    img = Image.open(f"{S}/frames/{f:06d}.jpg").convert("RGB"); W0, H0 = img.size
    z = np.load(f"{S}/output/omega_run/results_output/frame_{f}.npz"); zo = z["depth"].astype(np.float64); conf = z["conf"]
    H, W = zo.shape
    assert (W0, H0) == (W, H), (img.size, zo.shape)
    # PointDiT's own budget: 32x32 tokens of 16 px, aspect kept, multiples of 16
    fct = ((1024 * 256) / (H * W)) ** 0.5; nH = max(16, int(round(H * fct / 16)) * 16); nW = max(16, int(round(W * fct / 16)) * 16)
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    x = torch.nn.functional.interpolate(x, size=(nH, nW), mode="bilinear", align_corners=False)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        torch.cuda.synchronize(); t1 = time.time(); P = m.generate(x).float(); torch.cuda.synchronize(); dt = time.time() - t1
    P = torch.nn.functional.interpolate(P, size=(H, W), mode="nearest")[0].cpu().numpy()
    zm = P[2].astype(np.float64); nrm = np.linalg.norm(P, axis=0); valid_m = nrm <= 2.9
    valid = (zo > 0) & np.isfinite(zo) & (conf > 1e-3) & valid_m
    # robust affine z_omega ~ s*z_mono + b (Huber IRLS, 10 steps) over the valid pixels
    A = np.c_[zm[valid], np.ones(valid.sum())]; r = zo[valid]; w = np.ones(len(r))
    for _ in range(10):
        c = np.linalg.lstsq(A * w[:, None], r * w, rcond=None)[0]; e = r - A @ c; s_ = 1.4826 * np.median(np.abs(e)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.345 * s_ / np.maximum(np.abs(e), 1e-12)))
    za = c[0] * zm + c[1]
    rel = np.abs(za - zo) / np.maximum(zo, 1e-6)
    # edge sharpness: relative depth jump across 1 px, p99 of each map on the valid interior
    def grad_rel(d):
        gx = np.abs(np.diff(d, axis=1)) / np.maximum(d[:, 1:], 1e-6); gy = np.abs(np.diff(d, axis=0)) / np.maximum(d[1:, :], 1e-6)
        return np.percentile(gx[valid[:, 1:] & valid[:, :-1]], [50, 99]), np.percentile(gy[valid[1:, :] & valid[:-1, :]], [50, 99])
    print(f"frame {f}: pointdit {nW}x{nH} in {dt:.2f}s, VRAM peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB; valid(norm<=2.9) {valid_m.mean()*100:.1f} %; "
          f"affine s {c[0]:.3f} b {c[1]:.3f}; |rel| median {np.median(rel[valid])*100:.1f} % p90 {np.percentile(rel[valid],90)*100:.1f} %; "
          f"1-px rel jump p50/p99 omega {grad_rel(zo)} pointdit {grad_rel(za)}", flush=True)
    lo, hi = np.percentile(zo[valid], [2, 98])
    panel = np.concatenate([np.asarray(img), colour(np.where(valid, zo, np.nan), lo, hi), colour(np.where(valid, za, np.nan), lo, hi),
                            colour(np.where(valid, rel, np.nan), 0, 0.10)], axis=1)
    Image.fromarray(panel).save(f"{OUT}/pointdit_vs_omega_{f}.png")
    np.savez_compressed(f"{OUT}/pointdit_{f}.npz", z_mono=zm.astype(np.float32), valid=valid_m, s=c[0], b=c[1])
print("done", flush=True)
