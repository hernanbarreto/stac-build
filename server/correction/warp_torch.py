"""The epoch warp of the whole cloud on the device (USER 2026-11-11: "todo lo que me marcaste que puede pasarse
a GPU hay que hacerlo en GPU ... Nada debe hacerse en cpu").

``correction.apply.warp_full_cloud_numpy`` (the record) moves the cloud keyframe by keyframe on the CPU: the
depth along each point's own camera ray (z' = k·z + b) first, then the keyframe's rigid motion. Here every
point is moved at once on the evidence device: each point gathers its keyframe's (R, t, k, b, camera centre,
axis) and the same float64 arithmetic runs elementwise, the result cast back to the cloud's dtype as the
record's assignment does. The record's ``@`` is a BLAS product: the device's sums may round differently in
the last bit (measured on pccr 3108 against the record, tests/test_warp_torch.py).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def warp_full_cloud_t(session, R_kf: np.ndarray, t_kf: np.ndarray, k_kf: np.ndarray, device,
                      b_kf: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """``warp_full_cloud``: (xyz_new, n_moved) on ``device``."""
    dev = torch.device(device)
    n_kf = int(session.n_kf)
    R = np.asarray(R_kf, np.float64)[:n_kf]
    t = np.asarray(t_kf, np.float64)[:n_kf]
    kv = np.asarray(k_kf, np.float64)[:n_kf]
    bv = (np.asarray(b_kf, np.float64)[:n_kf] if b_kf is not None else np.zeros(n_kf))
    eye = np.eye(3)
    ident = (kv == 1.0) & (bv == 0.0) & np.array([np.allclose(R[k], eye) and np.allclose(t[k], 0.0)
                                                   for k in range(n_kf)], bool)
    cam = np.zeros((n_kf, 3)); axis = np.zeros((n_kf, 3))
    for k in range(n_kf):
        if kv[k] != 1.0 or bv[k] != 0.0:
            cam[k] = session.cam_center[session.frames[k]]
            axis[k] = session.poses[k][:3, 2]
    xyz_np = np.asarray(session.xyz)
    ks = torch.as_tensor(np.asarray(session.ks, np.int64), device=dev)
    inside = (ks >= 0) & (ks < n_kf)
    kk = torch.where(inside, ks, torch.zeros_like(ks))
    move = inside & ~torch.as_tensor(ident, device=dev)[kk]
    x = torch.as_tensor(xyz_np, device=dev)
    p = x[move].to(torch.float64)
    k_m = kk[move]
    Rt = torch.as_tensor(R, device=dev)[k_m]
    tt = torch.as_tensor(t, device=dev)[k_m]
    kvt = torch.as_tensor(kv, device=dev)[k_m]
    bvt = torch.as_tensor(bv, device=dev)[k_m]
    scale = (kvt != 1.0) | (bvt != 0.0)
    if bool(scale.any()):
        c = torch.as_tensor(cam, device=dev)[k_m]
        ax = torch.as_tensor(axis, device=dev)[k_m]
        d = p - c
        z = (d * ax).sum(1)
        zc = torch.where(z.abs() > 1e-9, z, torch.full_like(z, 1e-9))
        f = torch.where(bvt != 0.0, (kvt * z + bvt) / zc, kvt)
        p = torch.where(scale[:, None], c + d * f[:, None], p)
    p = (Rt @ p[:, :, None])[:, :, 0] + tt
    out = x.clone()
    out[move] = p.to(x.dtype)
    return out.cpu().numpy(), int(move.sum())
