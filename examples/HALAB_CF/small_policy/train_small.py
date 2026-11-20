#!/usr/bin/env python3
"""Small flow-matching policy on the cf_cache (Experiment 2 mechanism study).

A from-scratch CNN + state MLP conditions an MLP flow head that predicts the same
normalized 32x47 chunk as GR00T, with GR00T's flow convention
(x_t = (1-t) eps + t a, v = a - eps, t = 0.999 (1 - Beta(1.5, 1)), 4 Euler steps).
Images are 160x90 center crops of the 192x108 cache; rotation counterfactuals are
exact homographies applied on the GPU, with base-quaternion targets rotated by the
same angle.

Two batch modes:

* ``mix`` (SNR-law sweep): each sample drawn uniformly from the training episodes;
  a locomotion sample is rotated with probability ``--frac`` by delta ~ N(0, sigma).
* ``fixed`` (retention arms): every batch has exactly ``n_r`` rotated samples from
  the "old" episode pool (R), ``n_o`` plain old samples (O), ``n_e`` plain samples
  from the "new" pool (E) and optional extras.  Each role has its own RNG keyed by
  (seed, step), so for one seed the R samples, their angles, noise and flow times
  are identical in every arm.  Loss weight is 1/loss_norm per sample, so extra
  samples add gradient without rescaling the old part.

Optional distillation from a frozen teacher (the old model's EMA): nominal-output
(D), pair endpoints (E-arm) and pair response (F-arm), all on the velocity field.

Writes ``ckpt.pt`` (resumable; SIGTERM saves and exits with code 3), ``log.jsonl``
and, at the end, ``results.json`` with the evaluation of the EMA weights.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import halab_rotation_cf as rcf  # noqa: E402

ROOT = Path("/coc/flash2/jpark984/gr00t_baseline")
ACTION_KEYS = ("body_q", "base_quat", "inspire_hand", "camera")
STATE_KEYS = ("body_q", "body_dq", "base_quat", "inspire_hand", "camera")
QUAT_A = slice(29, 33)
QUAT_S = slice(58, 62)
GROUPS = {
    "legs": list(range(0, 12)),
    "waist": list(range(12, 15)),
    "arms": list(range(15, 29)),
    "root_quat": list(range(29, 33)),
    "hands": list(range(33, 45)),
    "camera": [45, 46],
}
H_IN, W_IN = 90, 160
HORIZON, ADIM, SDIM = 32, 47, 76
STOP = {"flag": False}


# ----------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/cf_cache")
    p.add_argument("--stats", type=Path,
                   default=ROOT / "outputs/halab_g1_47d_openhlm_b64_g256_lr1e4_v2/train_statistics.json")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--mode", choices=("mix", "fixed"), default="mix")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--min-lr-frac", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--ema", type=float, default=0.999)
    # rotation counterfactuals
    p.add_argument("--sigma-deg", type=float, default=0.0)
    p.add_argument("--max-deg", type=float, default=8.0)
    p.add_argument("--frac", type=float, default=1.0, help="mix mode: P(rotate | locomotion sample)")
    p.add_argument("--blur", type=int, default=1, help="downsample factor (1 = none)")
    p.add_argument("--gauge-deg", type=float, default=0.0,
                   help="random global yaw offset (uniform +-) on state and target quaternions")
    p.add_argument("--label-sign", type=int, default=0, help="0 = read cache calibration.json")
    # fixed-mode composition
    p.add_argument("--n-r", type=int, default=32)
    p.add_argument("--n-o", type=int, default=96)
    p.add_argument("--n-e", type=int, default=0)
    p.add_argument("--n-xe", type=int, default=0)
    p.add_argument("--n-xo", type=int, default=0)
    p.add_argument("--loss-norm", type=float, default=128.0)
    p.add_argument("--stream-offset", type=int, default=0,
                   help="added to the step in every sampling key (continuations use fresh draws)")
    p.add_argument("--pool-seed", type=int, default=11)
    p.add_argument("--old-frac", type=float, default=0.5)
    # continuation / distillation
    p.add_argument("--init-from", type=Path, help="run dir whose final ckpt.pt initializes model, EMA, optimizer")
    p.add_argument("--teacher-from", type=Path, help="run dir whose EMA is the frozen teacher (default init-from)")
    p.add_argument("--distill-nominal", type=float, default=0.0)
    p.add_argument("--distill-endpoints", type=float, default=0.0)
    p.add_argument("--distill-response", type=float, default=0.0)
    p.add_argument("--n-distill", type=int, default=32)
    p.add_argument("--n-pairs", type=int, default=16)
    # bookkeeping
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--gain-anchors", type=int, default=768)
    p.add_argument("--nominal-anchors", type=int, default=2048)
    p.add_argument("--eval-seeds", type=int, default=4)
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Cache, pools, normalization
# ----------------------------------------------------------------------------


class CacheIndex:
    """Light metadata (no images) shared by the main process and the workers."""

    def __init__(self, cache: Path, pool_seed: int, old_frac: float) -> None:
        meta = json.loads((cache / "meta.json").read_text())
        self.cache = cache
        self.meta = meta
        eps = meta["episodes"]
        self.row_offset = np.asarray([e["row_offset"] for e in eps], dtype=np.int64)
        self.length = np.asarray([e["length"] for e in eps], dtype=np.int64)
        self.frames_ep = np.load(cache / "frames_ep.npy")
        self.frames_row = np.load(cache / "frames_row.npy").astype(np.int64)
        self.frames_walk = np.load(cache / "frames_walk.npy").astype(bool)
        self.frames_up = np.load(cache / "frames_up.npy").astype(np.float32)
        split = np.asarray([e["split"] for e in eps])
        is_train = split[self.frames_ep] == "train"
        self.train = np.where(is_train)[0]
        self.test = np.where(~is_train)[0]
        # Task-stratified old/new pools inside the training episodes.
        by_task: dict[str, list[int]] = {}
        for i, e in enumerate(eps):
            if e["split"] == "train":
                by_task.setdefault(e["task"], []).append(i)
        old_eps = set()
        rng = random.Random(pool_seed)
        for task in sorted(by_task):
            ids = sorted(by_task[task], key=lambda i: eps[i]["episode_index"])
            rng.shuffle(ids)
            old_eps.update(ids[: int(round(old_frac * len(ids)))])
        is_old = np.isin(self.frames_ep, sorted(old_eps)) & is_train
        is_new = is_train & ~is_old
        self.old = np.where(is_old)[0]
        self.new = np.where(is_new)[0]
        self.old_walk = np.where(is_old & self.frames_walk)[0]
        self.train_walk = np.where(is_train & self.frames_walk)[0]
        self.test_walk = np.where(~is_train & self.frames_walk)[0]
        self.old_keys = sorted(eps[i]["key"] for i in old_eps)

    def chunk_rows(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ep = self.frames_ep[ids]
        row = self.frames_row[ids]
        off = self.row_offset[ep]
        length = self.length[ep]
        steps = row[:, None] + 1 + np.arange(HORIZON)[None, :]
        mask = (steps < length[:, None]).astype(np.float32)
        rows = off[:, None] + np.minimum(steps, (length - 1)[:, None])
        return off + row, rows, mask


def load_norm(stats_path: Path) -> dict[str, np.ndarray]:
    stats = json.loads(stats_path.read_text())
    emb = next(iter(stats))
    out = {}
    for kind, keys in (("action", ACTION_KEYS), ("state", STATE_KEYS)):
        lo = np.concatenate([np.asarray(stats[emb][kind][k]["q01"], dtype=np.float64) for k in keys])
        hi = np.concatenate([np.asarray(stats[emb][kind][k]["q99"], dtype=np.float64) for k in keys])
        out[f"{kind}_c"] = ((hi + lo) / 2).astype(np.float32)
        out[f"{kind}_h"] = np.maximum((hi - lo) / 2, 1e-6).astype(np.float32)
    assert out["action_c"].shape == (ADIM,) and out["state_c"].shape == (SDIM,)
    return out


# ----------------------------------------------------------------------------
# Batch specs (deterministic per step) and loading
# ----------------------------------------------------------------------------


ROLE_R, ROLE_O, ROLE_E, ROLE_XE, ROLE_XO = range(5)


def _flow_noise(key, n):
    rng = np.random.default_rng(key)
    eps = rng.standard_normal((n, HORIZON, ADIM)).astype(np.float32)
    t = (0.999 * (1.0 - rng.beta(1.5, 1.0, size=n))).astype(np.float32)
    return eps, t


def batch_spec(args, idx: CacheIndex, step: int) -> dict:
    s = args.seed
    step = int(step) + int(args.stream_offset)
    sig, lim = math.radians(args.sigma_deg), math.radians(args.max_deg)
    if args.mode == "mix":
        rng = np.random.default_rng([s, 11, step])
        ids = rng.choice(idx.train, args.batch)
        rot = idx.frames_walk[ids] & (rng.random(args.batch) < args.frac) & (sig > 0)
        delta = np.where(rot, np.clip(rng.normal(0.0, max(sig, 1e-9), args.batch), -lim, lim), 0.0)
        gauge = (rng.uniform(-1, 1, args.batch) * math.radians(args.gauge_deg)) if args.gauge_deg > 0 else np.zeros(args.batch)
        eps, t = _flow_noise([s, 30, step], args.batch)
        spec = {"ids": ids, "delta": delta, "gauge": gauge, "eps": eps, "t": t,
                "weight": np.full(args.batch, 1.0 / args.batch), "role": np.zeros(args.batch, dtype=np.int64)}
    else:
        parts = []
        rng = np.random.default_rng([s, 12, step])
        ids_r = rng.choice(idx.old_walk, 32)
        d_r = np.clip(rng.normal(0.0, max(sig, 1e-9), 32), -lim, lim) if sig > 0 else np.zeros(32)
        e_r, t_r = _flow_noise([s, 31, step], 32)
        parts.append((ids_r[: args.n_r], d_r[: args.n_r], e_r[: args.n_r], t_r[: args.n_r], ROLE_R))
        for role, pool, n_draw, n_use, k in (
            (ROLE_O, idx.old, 96, args.n_o, 13),
            (ROLE_E, idx.new, 64, args.n_e, 14),
            (ROLE_XE, idx.new, 48, args.n_xe, 15),
            (ROLE_XO, idx.old, 48, args.n_xo, 16),
        ):
            if n_use <= 0:
                continue
            ids = np.random.default_rng([s, k, step]).choice(pool, n_draw)[:n_use]
            e, t = _flow_noise([s, k + 20, step], n_draw)
            parts.append((ids, np.zeros(n_use), e[:n_use], t[:n_use], role))
        ids = np.concatenate([p[0] for p in parts])
        spec = {"ids": ids, "delta": np.concatenate([p[1] for p in parts]),
                "gauge": np.zeros(len(ids)), "eps": np.concatenate([p[2] for p in parts]),
                "t": np.concatenate([p[3] for p in parts]),
                "weight": np.full(len(ids), 1.0 / args.loss_norm),
                "role": np.concatenate([np.full(len(p[0]), p[4], dtype=np.int64) for p in parts])}
    if args.distill_nominal > 0:
        spec["d_ids"] = np.random.default_rng([s, 17, step]).choice(idx.old, args.n_distill)
        spec["d_eps"], spec["d_t"] = _flow_noise([s, 36, step], args.n_distill)
    if args.distill_endpoints > 0 or args.distill_response > 0:
        rng = np.random.default_rng([s, 18, step])
        spec["p_ids"] = rng.choice(idx.old_walk, args.n_pairs)
        spec["p_delta"] = rng.choice([-1.0, 1.0], args.n_pairs) * np.radians(rng.uniform(3.0, 6.0, args.n_pairs))
        spec["p_eps"], spec["p_t"] = _flow_noise([s, 37, step], args.n_pairs)
    return spec


class StepDataset(Dataset):
    def __init__(self, args, idx: CacheIndex) -> None:
        self.args, self.idx = args, idx
        self.frames = self.rows_state = self.rows_action = None

    def _open(self):
        if self.frames is None:
            c = self.idx.cache
            self.frames = np.load(c / "frames.npy", mmap_mode="r")
            self.rows_state = np.load(c / "rows_state.npy", mmap_mode="r")
            self.rows_action = np.load(c / "rows_action.npy", mmap_mode="r")

    def load(self, ids: np.ndarray) -> dict:
        self._open()
        srow, rows, mask = self.idx.chunk_rows(ids)
        return {
            "img": np.ascontiguousarray(self.frames[ids]),
            "state": np.ascontiguousarray(self.rows_state[srow]),
            "chunk": np.ascontiguousarray(self.rows_action[rows]),
            "mask": mask,
            "up": self.idx.frames_up[ids],
        }

    def __len__(self) -> int:
        return self.args.steps

    def __getitem__(self, step: int) -> dict:
        spec = batch_spec(self.args, self.idx, int(step))
        out = {"step": int(step), "main": self.load(spec["ids"])}
        out["main"].update({k: spec[k] for k in ("delta", "gauge", "eps", "t", "weight", "role")})
        if "d_ids" in spec:
            out["dist"] = self.load(spec["d_ids"])
            out["dist"].update({"eps": spec["d_eps"], "t": spec["d_t"]})
        if "p_ids" in spec:
            out["pair"] = self.load(spec["p_ids"])
            out["pair"].update({"delta": spec["p_delta"], "eps": spec["p_eps"], "t": spec["p_t"]})
        return out


class RangeSampler(Sampler):
    def __init__(self, start: int, stop: int) -> None:
        self.start, self.stop = start, stop

    def __iter__(self):
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return max(0, self.stop - self.start)


# ----------------------------------------------------------------------------
# GPU transforms
# ----------------------------------------------------------------------------


class Geometry:
    def __init__(self, height: int, width: int, device) -> None:
        k = torch.tensor(rcf.intrinsics(width, height), dtype=torch.float32, device=device)
        self.k, self.k_inv = k, torch.linalg.inv(k)
        vs, us = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32),
                                torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        self.pix = torch.stack([us, vs, torch.ones_like(us)], dim=-1).reshape(-1, 3)  # (HW, 3)
        self.h, self.w = height, width

    def warp(self, img: torch.Tensor, up: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """img (n,3,H,W) float, up (n,3), delta (n,) rad: scene rotation counterfactual."""

        x, y, z = up[:, 0], up[:, 1], up[:, 2]
        zero = torch.zeros_like(x)
        kx = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(-1, 3, 3)
        eye = torch.eye(3, device=img.device).expand_as(kx)
        s, c = torch.sin(delta)[:, None, None], torch.cos(delta)[:, None, None]
        rot = eye + s * kx + (1 - c) * (kx @ kx)
        hom = self.k @ rot @ self.k_inv
        src = (torch.linalg.inv(hom) @ self.pix.T).transpose(1, 2)  # (n, HW, 3)
        xs = src[..., 0] / src[..., 2]
        ys = src[..., 1] / src[..., 2]
        grid = torch.stack([2 * xs / (self.w - 1) - 1, 2 * ys / (self.h - 1) - 1], dim=-1)
        grid = grid.reshape(-1, self.h, self.w, 2)
        return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def quat_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def yaw_rotate_t(q: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """World-yaw rotation of wxyz quaternions q (n, ..., 4) by angle (n,)."""

    half = angle / 2
    qz = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=-1)
    while qz.dim() < q.dim():
        qz = qz.unsqueeze(1)
    return quat_mul_t(qz.expand_as(q), q)


def yaw_of_t(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_t(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class Prep:
    """Uint8 cache frames -> normalized 160x90 inputs; raw targets -> normalized."""

    def __init__(self, norm: dict, args, cache_hw, device, label_sign: int) -> None:
        self.geo = Geometry(cache_hw[0], cache_hw[1], device)
        self.dev = device
        self.a_c = torch.tensor(norm["action_c"], device=device)
        self.a_h = torch.tensor(norm["action_h"], device=device)
        self.s_c = torch.tensor(norm["state_c"], device=device)
        self.s_h = torch.tensor(norm["state_h"], device=device)
        self.blur = int(args.blur)
        self.sign = float(label_sign)
        self.y0 = (cache_hw[0] - H_IN) // 2
        self.x0 = (cache_hw[1] - W_IN) // 2

    def images(self, img_u8: torch.Tensor, up: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = img_u8.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
        moved = delta.abs() > 0
        if bool(moved.any()):
            x = x.clone()
            x[moved] = self.geo.warp(x[moved], up[moved], delta[moved])
        x = x[:, :, self.y0 : self.y0 + H_IN, self.x0 : self.x0 + W_IN]
        if self.blur > 1:
            x = F.interpolate(x, scale_factor=1.0 / self.blur, mode="area")
            x = F.interpolate(x, size=(H_IN, W_IN), mode="bilinear", align_corners=False)
        return (x - 0.5) / 0.25

    def targets(self, chunk: torch.Tensor, state: torch.Tensor, delta: torch.Tensor, gauge: torch.Tensor):
        chunk = chunk.to(self.dev).float().clone()
        state = state.to(self.dev).float().clone()
        yaw = self.sign * delta + gauge
        if bool((yaw.abs() > 0).any()):
            chunk[..., QUAT_A] = yaw_rotate_t(chunk[..., QUAT_A], yaw)
        if bool((gauge.abs() > 0).any()):
            state[:, QUAT_S] = yaw_rotate_t(state[:, QUAT_S], gauge)
        a_n = ((chunk - self.a_c) / self.a_h).clamp(-1, 1)
        s_n = ((state - self.s_c) / self.s_h).clamp(-3, 3)
        return a_n, s_n, chunk, state

    def decode(self, a_n: torch.Tensor) -> torch.Tensor:
        return a_n * self.a_h + self.a_c


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(y + (x if self.skip is None else self.skip(x)))


class Encoder(nn.Module):
    """160x90 -> 6x10 feature map -> 64 spatial-softmax keypoints + pooled features."""

    def __init__(self, keypoints: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 5, 2, 2, bias=False), nn.GroupNorm(8, 32), nn.SiLU())
        self.body = nn.Sequential(
            Block(32, 64, 2), Block(64, 64), Block(64, 128, 2), Block(128, 128), Block(128, 256, 2), Block(256, 256))
        self.kp = nn.Conv2d(256, keypoints, 1)
        self.out = nn.Sequential(nn.Linear(2 * keypoints + 256, 512), nn.SiLU())

    def forward(self, x):
        f = self.body(self.stem(x))
        n, _, h, w = f.shape
        logits = self.kp(f).flatten(2)  # (n, K, hw)
        att = logits.softmax(-1)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        kx = (att * gx.reshape(1, 1, -1)).sum(-1)
        ky = (att * gy.reshape(1, 1, -1)).sum(-1)
        feats = torch.cat([kx, ky, f.mean(dim=(2, 3))], dim=-1)
        return self.out(feats)


def time_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = (t.float() * 1000.0)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResFiLM(nn.Module):
    def __init__(self, width: int, cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(cond, 2 * width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=-1)
        y = self.norm(h) * (1 + scale) + shift
        return h + self.fc2(F.silu(self.fc1(F.silu(y))))


class FlowPolicy(nn.Module):
    def __init__(self, width: int = 1024, depth: int = 4) -> None:
        super().__init__()
        self.enc = Encoder()
        self.state = nn.Sequential(nn.Linear(SDIM, 256), nn.SiLU(), nn.Linear(256, 256))
        self.cond = nn.Sequential(nn.Linear(512 + 256, 512), nn.SiLU())
        self.temb = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 512))
        self.inp = nn.Linear(HORIZON * ADIM, width)
        self.blocks = nn.ModuleList([ResFiLM(width, 512) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, HORIZON * ADIM)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def encode(self, img, state_n):
        f = self.enc(img)
        return self.cond(torch.cat([f, self.state(state_n)], dim=-1)), f

    def velocity(self, cond, x_t, t):
        c = cond + self.temb(time_embedding(t))
        h = self.inp(x_t.flatten(1))
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.out_norm(h)).reshape(-1, HORIZON, ADIM)

    def forward(self, img, state_n, x_t, t):
        cond, _ = self.encode(img, state_n)
        return self.velocity(cond, x_t, t)

    @torch.no_grad()
    def sample(self, img, state_n, eps, steps: int = 4):
        cond, _ = self.encode(img, state_n)
        x = eps
        for i in range(steps):
            t = torch.full((x.shape[0],), i / steps, device=x.device)
            x = x + self.velocity(cond, x, t).float() / steps
        return x


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------


def evenly(ids: np.ndarray, n: int) -> np.ndarray:
    if len(ids) <= n:
        return ids
    return ids[np.linspace(0, len(ids) - 1, n).round().astype(int)]


@torch.no_grad()
def evaluate(model: FlowPolicy, args, idx: CacheIndex, prep: Prep, label_sign: int) -> dict:
    model.eval()
    dev = prep.dev
    ds = StepDataset(args, idx)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    h_a = prep.a_h

    def predict(batch, delta, yaw_offset, eps):
        n = len(batch["state"])
        d = torch.full((n,), float(delta), device=dev)
        img = prep.images(torch.from_numpy(batch["img"]), torch.from_numpy(batch["up"]).to(dev), d)
        state = torch.from_numpy(batch["state"]).to(dev).float().clone()
        if yaw_offset != 0.0:
            state[:, QUAT_S] = yaw_rotate_t(state[:, QUAT_S], torch.full((n,), float(yaw_offset), device=dev))
        s_n = ((state - prep.s_c) / prep.s_h).clamp(-3, 3)
        with amp:
            pred_n = model.sample(img, s_n, eps)
        return pred_n.float()

    out: dict = {}
    # Nominal fidelity and turning slope on held-out (val) episodes.
    nom = evenly(idx.test, args.nominal_anchors)
    abs_sum = torch.zeros(ADIM, device=dev)
    n_valid = 0.0
    pred_d, gt_d = [], []
    for b in range(0, len(nom), 256):
        batch = ds.load(nom[b : b + 256])
        n = len(batch["state"])
        gen = torch.Generator(device=dev).manual_seed(5000 + b)
        eps = torch.randn((n, HORIZON, ADIM), generator=gen, device=dev)
        pred_n = predict(batch, 0.0, 0.0, eps)
        a_n, _, chunk, state = prep.targets(torch.from_numpy(batch["chunk"]), torch.from_numpy(batch["state"]),
                                            torch.zeros(n, device=dev), torch.zeros(n, device=dev))
        mask = torch.from_numpy(batch["mask"]).to(dev)
        abs_sum += ((pred_n - a_n).abs() * mask[..., None]).sum(dim=(0, 1))
        n_valid += float(mask.sum())
        yaw_state = yaw_of_t(state[:, QUAT_S])
        yaw_pred = yaw_of_t(prep.decode(pred_n)[..., QUAT_A])
        yaw_gt = yaw_of_t(chunk[..., QUAT_A])
        ok = mask[:, -1] > 0
        pred_d.append(wrap_t(yaw_pred[:, -1] - yaw_state)[ok].cpu())
        gt_d.append(wrap_t(yaw_gt[:, -1] - yaw_state)[ok].cpu())
    mae_dim = (abs_sum / max(n_valid, 1.0)) * h_a
    out["nominal_mae_denorm"] = float(mae_dim.mean())
    for g, dims in GROUPS.items():
        out[f"nominal_mae_denorm/{g}"] = float(mae_dim[dims].mean())
    pd_, gd_ = torch.cat(pred_d).numpy(), torch.cat(gt_d).numpy()
    out["turn_slope_origin"] = float((pd_ * gd_).sum() / max((gd_**2).sum(), 1e-12))
    out["turn_corr"] = float(np.corrcoef(gd_, pd_)[0, 1]) if len(gd_) > 2 else float("nan")

    # Visual heading gain, state-yaw gain, off-target response.
    gain = evenly(idx.test_walk, args.gain_anchors)
    deltas = [math.radians(d) for d in (-6.0, -3.0, 3.0, 6.0)]
    alphas = [math.radians(10.0), math.radians(-10.0)]
    num_v = torch.zeros(HORIZON, device=dev)
    num_p = torch.zeros(HORIZON, device=dev)
    num_pos = num_neg = 0.0
    den_v = den_p = den_pos = den_neg = 0.0
    off_sum, off_n = 0.0, 0
    for b in range(0, len(gain), 256):
        batch = ds.load(gain[b : b + 256])
        n = len(batch["state"])
        for s in range(args.eval_seeds):
            gen = torch.Generator(device=dev).manual_seed(9000 + 97 * s + b)
            eps = torch.randn((n, HORIZON, ADIM), generator=gen, device=dev)
            ref = prep.decode(predict(batch, 0.0, 0.0, eps))
            ref_yaw = yaw_of_t(ref[..., QUAT_A])
            for d in deltas:
                p = prep.decode(predict(batch, d, 0.0, eps))
                dyaw = wrap_t(yaw_of_t(p[..., QUAT_A]) - ref_yaw)
                x = label_sign * d
                num_v += (x * dyaw).sum(0)
                den_v += n * d * d
                if d > 0:
                    num_pos += float((x * dyaw[:, -1]).sum())
                    den_pos += n * d * d
                else:
                    num_neg += float((x * dyaw[:, -1]).sum())
                    den_neg += n * d * d
                off_sum += float(((p[..., :29] - ref[..., :29]).abs().mean(dim=(1, 2)) / abs(d)).sum())
                off_n += n
            for a in alphas:
                p = prep.decode(predict(batch, 0.0, a, eps))
                dyaw = wrap_t(yaw_of_t(p[..., QUAT_A]) - ref_yaw)
                num_p += (a * dyaw).sum(0)
                den_p += n * a * a
    gv = (num_v / max(den_v, 1e-12)).cpu().numpy()
    gp = (num_p / max(den_p, 1e-12)).cpu().numpy()
    out.update({
        "G_v_row": gv.tolist(), "G_v_0": float(gv[0]), "G_v_15": float(gv[15]), "G_v_31": float(gv[31]),
        "G_v_mean": float(gv.mean()), "G_v_31_pos": num_pos / max(den_pos, 1e-12),
        "G_v_31_neg": num_neg / max(den_neg, 1e-12),
        "G_p_row": gp.tolist(), "G_p_0": float(gp[0]), "G_p_31": float(gp[31]),
        "offtarget_body_rad_per_rad": off_sum / max(off_n, 1),
        "n_gain_anchors": int(len(gain)), "n_nominal_anchors": int(len(nom)),
    })

    # Linear probe: how well do image features encode the applied rotation? (sqrt(N) in deg)
    rng = np.random.default_rng(123)
    feats, targ = [], []
    for b in range(0, len(gain), 256):
        batch = ds.load(gain[b : b + 256])
        n = len(batch["state"])
        d = rng.uniform(-6.0, 6.0, n)
        img = prep.images(torch.from_numpy(batch["img"]), torch.from_numpy(batch["up"]).to(dev),
                          torch.tensor(np.radians(d), dtype=torch.float32, device=dev))
        with amp:
            f = model.enc(img)
        feats.append(f.float().cpu().numpy())
        targ.append(d)
    xf, yf = np.concatenate(feats), np.concatenate(targ)
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    xs = (xf - mu) / sd
    tr, te = np.arange(len(yf)) % 2 == 0, np.arange(len(yf)) % 2 == 1
    xtr = np.concatenate([xs[tr], np.ones((tr.sum(), 1))], 1)
    xte = np.concatenate([xs[te], np.ones((te.sum(), 1))], 1)
    w = np.linalg.solve(xtr.T @ xtr + 1.0 * np.eye(xtr.shape[1]), xtr.T @ yf[tr])
    resid = yf[te] - xte @ w
    out["probe_rotation_resid_deg"] = float(resid.std())
    out["probe_rotation_r2"] = float(1 - resid.var() / max(yf[te].var(), 1e-12))
    model.train()
    return out


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def lr_at(args, step: int) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    prog = (step - args.warmup) / max(args.steps - args.warmup, 1)
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))


def save_ckpt(path: Path, model, ema, opt, step: int, args) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                "step": step, "args": {k: str(v) for k, v in vars(args).items()}}, tmp)
    os.replace(tmp, path)


def velocity_loss(v, target, mask):
    err = ((v.float() - target) ** 2 * mask[..., None]).sum(dim=(1, 2))
    return err / (mask.sum(dim=1) * ADIM).clamp_min(1.0)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "results.json").is_file():
        print(f"{args.run_dir} already complete", flush=True)
        return 0
    signal.signal(signal.SIGTERM, lambda *_: STOP.update(flag=True))
    torch.manual_seed(args.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    label_sign = args.label_sign
    if label_sign == 0:
        label_sign = int(json.loads((args.cache / "calibration.json").read_text())["label_sign"])
    if label_sign not in (1, -1):
        if args.sigma_deg > 0 or args.distill_endpoints > 0 or args.distill_response > 0:
            raise SystemExit(f"warp calibration failed (label_sign={label_sign}); refusing rotation labels")
        label_sign = 1

    idx = CacheIndex(args.cache, args.pool_seed, args.old_frac)
    meta = idx.meta
    norm = load_norm(args.stats)
    prep = Prep(norm, args, (meta["height"], meta["width"]), dev, label_sign)
    (args.run_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "label_sign_used": label_sign,
         "n_train": int(len(idx.train)), "n_old": int(len(idx.old)), "n_new": int(len(idx.new)),
         "n_old_walk": int(len(idx.old_walk)), "n_test": int(len(idx.test)),
         "old_episode_keys": idx.old_keys}, indent=1) + "\n")

    model = FlowPolicy().to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    ckpt_path = args.run_dir / "ckpt.pt"
    start = 0
    if ckpt_path.is_file():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        print(f"resumed {ckpt_path} at step {start}", flush=True)
    elif args.init_from is not None:
        ck = torch.load(args.init_from / "ckpt.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        print(f"initialized from {args.init_from} (its step {ck['step']})", flush=True)

    teacher = None
    if args.distill_nominal > 0 or args.distill_endpoints > 0 or args.distill_response > 0:
        src = args.teacher_from or args.init_from
        if src is None:
            raise SystemExit("distillation needs --teacher-from or --init-from")
        teacher = FlowPolicy().to(dev).eval()
        teacher.load_state_dict(torch.load(src / "ckpt.pt", map_location=dev, weights_only=False)["ema"])
        for p in teacher.parameters():
            p.requires_grad_(False)

    loader = DataLoader(StepDataset(args, idx), batch_size=None, sampler=RangeSampler(start, args.steps),
                        num_workers=args.workers, pin_memory=True, prefetch_factor=4 if args.workers > 0 else None,
                        persistent_workers=False)
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    log = open(args.run_dir / "log.jsonl", "a")
    t_start, t_last = time.time(), time.time()
    run_loss, run_n = 0.0, 0
    final_step = start
    for batch in loader:
        step = int(batch["step"])
        final_step = step + 1
        for g in opt.param_groups:
            g["lr"] = lr_at(args, step)
        m = batch["main"]
        delta = m["delta"].to(dev).float()
        gauge = m["gauge"].to(dev).float()
        up = m["up"].to(dev).float()
        img = prep.images(m["img"], up, delta)
        a_n, s_n, _, _ = prep.targets(m["chunk"], m["state"], delta, gauge)
        mask = m["mask"].to(dev).float()
        eps = m["eps"].to(dev).float()
        t = m["t"].to(dev).float()
        x_t = (1 - t)[:, None, None] * eps + t[:, None, None] * a_n
        with amp:
            v = model(img, s_n, x_t, t)
        loss = (m["weight"].to(dev).float() * velocity_loss(v, a_n - eps, mask)).sum()
        logs = {"fm": float(loss)}

        if teacher is not None and "dist" in batch:
            d = batch["dist"]
            n = len(d["state"])
            zero = torch.zeros(n, device=dev)
            d_img = prep.images(d["img"], d["up"].to(dev).float(), zero)
            d_a, d_s, _, _ = prep.targets(d["chunk"], d["state"], zero, zero)
            d_eps, d_t = d["eps"].to(dev).float(), d["t"].to(dev).float()
            d_x = (1 - d_t)[:, None, None] * d_eps + d_t[:, None, None] * d_a
            with amp:
                v_s = model(d_img, d_s, d_x, d_t)
                with torch.no_grad():
                    v_t = teacher(d_img, d_s, d_x, d_t)
            l_d = ((v_s.float() - v_t.float()) ** 2).mean()
            loss = loss + args.distill_nominal * l_d
            logs["distill_nominal"] = float(l_d)
        if teacher is not None and "pair" in batch:
            pr = batch["pair"]
            n = len(pr["state"])
            zero = torch.zeros(n, device=dev)
            p_delta = pr["delta"].to(dev).float()
            p_up = pr["up"].to(dev).float()
            img_minus = prep.images(pr["img"], p_up, zero)
            img_plus = prep.images(pr["img"], p_up, p_delta)
            p_a, p_s, _, _ = prep.targets(pr["chunk"], pr["state"], zero, zero)
            p_eps, p_t = pr["eps"].to(dev).float(), pr["t"].to(dev).float()
            p_x = (1 - p_t)[:, None, None] * p_eps + p_t[:, None, None] * p_a
            imgs = torch.cat([img_minus, img_plus])
            ss = torch.cat([p_s, p_s])
            xx = torch.cat([p_x, p_x])
            tt = torch.cat([p_t, p_t])
            with amp:
                v_s = model(imgs, ss, xx, tt).float()
                with torch.no_grad():
                    v_t = teacher(imgs, ss, xx, tt).float()
            vs_m, vs_p = v_s[:n], v_s[n:]
            vt_m, vt_p = v_t[:n], v_t[n:]
            if args.distill_endpoints > 0:
                l_e = ((vs_m - vt_m) ** 2).mean() + ((vs_p - vt_p) ** 2).mean()
                loss = loss + args.distill_endpoints * l_e
                logs["distill_endpoints"] = float(l_e)
            if args.distill_response > 0:
                l_f = (((vs_p - vs_m) - (vt_p - vt_m)) ** 2).mean()
                loss = loss + args.distill_response * l_f
                logs["distill_response"] = float(l_f)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run_loss += float(loss)
        run_n += 1
        if (step + 1) % args.log_every == 0:
            now = time.time()
            rec = {"step": step + 1, "loss": run_loss / run_n, "lr": lr_at(args, step), "grad_norm": float(gnorm),
                   "s_per_step": (now - t_last) / args.log_every, **logs}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            run_loss, run_n, t_last = 0.0, 0, now
        if (step + 1) % args.save_every == 0 or STOP["flag"]:
            save_ckpt(ckpt_path, model, ema, opt, step + 1, args)
        if STOP["flag"]:
            print(f"SIGTERM: saved at step {step + 1}; exiting for resubmission", flush=True)
            return 3
    save_ckpt(ckpt_path, model, ema, opt, final_step, args)
    t_eval = time.time()
    metrics = evaluate(ema, args, idx, prep, label_sign)
    result = {"run_dir": str(args.run_dir), "steps": final_step, "label_sign": label_sign,
              "train_seconds": t_eval - t_start, "eval_seconds": time.time() - t_eval, "eval": metrics,
              "config": {k: str(v) for k, v in vars(args).items()}}
    (args.run_dir / "results.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in metrics.items() if not k.endswith("_row")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
