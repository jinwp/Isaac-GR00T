#!/usr/bin/env python3
"""Summarize Experiment 2 (and the probes of Experiment 1) into summary.md + CSVs.

* SNR law: fit G = G_max * S / (S + N) with S = S0 + f * E[delta^2] on the signal
  arms (f = 1, sigma 0/2/4/6), then predict the held-out arms: matched-exposure
  dilution (f = 0.5, 0.25) directly, and blur with N scaled by the linear-probe
  residual ratio.
* Retention: per arm mean +- sd over seeds, and paired differences vs arm A.
  The SNR law predicts A ~ C, B < A, H ~ H2 < A.  The ChatGPT report's H1
  (fluent data damages the response beyond exposure) predicts C < A and H < H2.
* Probes: G_v, G_p and the turning slope for each GR00T checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

import numpy as np


OUT = Path("/coc/flash2/jpark984/gr00t_baseline/outputs/cf_exp2")
KEYS = ("G_v_mean", "G_v_0", "G_v_31", "G_p_31", "offtarget_body_rad_per_rad", "nominal_mae_denorm",
        "turn_slope_origin", "probe_rotation_resid_deg")


def load_runs() -> dict[str, dict[int, dict]]:
    arms: dict[str, dict[int, dict]] = {}
    for path in sorted((OUT / "runs").glob("*/results.json")):
        m = re.match(r"(.+)_s(\d+)$", path.parent.name)
        if not m:
            continue
        arms.setdefault(m.group(1), {})[int(m.group(2))] = json.loads(path.read_text())["eval"]
    return arms


def mean_sd(values) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(m: float, s: float, digits: int = 3) -> str:
    return "n/a" if not np.isfinite(m) else f"{m:.{digits}f} ± {s:.{digits}f}"


def clipped_var(sigma_deg: float, max_deg: float = 8.0) -> float:
    if sigma_deg <= 0:
        return 0.0
    rng = np.random.default_rng(0)
    d = np.clip(rng.normal(0.0, math.radians(sigma_deg), 1_000_000), -math.radians(max_deg), math.radians(max_deg))
    return float((d * d).mean())


def fit_snr(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """points: (signal variance rad^2, observed G). Returns (G_max, S0, N, rmse)."""

    best = (float("nan"),) * 4
    best_err = float("inf")
    s_obs = np.asarray([p[0] for p in points])
    g_obs = np.asarray([p[1] for p in points])
    for s0 in np.concatenate([[0.0], np.logspace(-7, -1, 121)]):
        for n in np.logspace(-6, 0, 241):
            basis = (s0 + s_obs) / (s0 + s_obs + n)
            gmax = float((basis * g_obs).sum() / max((basis * basis).sum(), 1e-12))
            err = float(np.sqrt(((gmax * basis - g_obs) ** 2).mean()))
            if err < best_err:
                best_err, best = err, (gmax, float(s0), float(n), err)
    return best


def main() -> None:
    arms = load_runs()
    lines = ["# Experiment 2 summary", ""]
    rows = []
    for arm in sorted(arms):
        rec = {"arm": arm, "seeds": len(arms[arm])}
        for k in KEYS:
            rec[k], rec[k + "_sd"] = mean_sd([arms[arm][s].get(k) for s in arms[arm]])
        rows.append(rec)
    if rows:
        with open(OUT / "arms.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    table = {r["arm"]: r for r in rows}

    lines += ["## All arms (mean ± sd over seeds)", "",
              "| arm | seeds | G_v mean | G_v row 31 | G_p row 31 | nominal MAE | turn slope | probe resid (deg) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['seeds']} | {fmt(r['G_v_mean'], r['G_v_mean_sd'])} | "
                     f"{fmt(r['G_v_31'], r['G_v_31_sd'])} | {fmt(r['G_p_31'], r['G_p_31_sd'])} | "
                     f"{fmt(r['nominal_mae_denorm'], r['nominal_mae_denorm_sd'], 4)} | "
                     f"{fmt(r['turn_slope_origin'], r['turn_slope_origin_sd'])} | "
                     f"{fmt(r['probe_rotation_resid_deg'], r['probe_rotation_resid_deg_sd'], 2)} |")
    lines.append("")

    signal = {"snr_s0": 0.0, "snr_s2": 2.0, "snr_s4": 4.0, "snr_s6": 6.0}
    pts = [(clipped_var(sig), table[a]["G_v_mean"]) for a, sig in signal.items()
           if a in table and np.isfinite(table[a]["G_v_mean"])]
    lines += ["## SNR law", ""]
    if len(pts) >= 3:
        gmax, s0, n, rmse = fit_snr(pts)
        lines.append(f"Fit on the signal arms: G_max = {gmax:.3f}, sqrt(S0) = {math.degrees(math.sqrt(s0)):.2f} deg, "
                     f"sqrt(N) = {math.degrees(math.sqrt(n)):.2f} deg, rmse = {rmse:.4f}.")
        lines += ["", "| held-out arm | predicted G_v mean | observed |", "| --- | --- | --- |"]
        v4 = clipped_var(4.0)
        for arm, f in (("snr_s4_f050_matched", 0.5), ("snr_s4_f025_matched", 0.25),
                       ("snr_s4_f050_fixed", 0.5), ("snr_s4_f025_fixed", 0.25)):
            if arm in table:
                pred = gmax * (s0 + f * v4) / (s0 + f * v4 + n)
                note = " (exposure also diluted)" if arm.endswith("fixed") else ""
                lines.append(f"| {arm}{note} | {pred:.3f} | {fmt(table[arm]['G_v_mean'], table[arm]['G_v_mean_sd'])} |")
        base = table.get("snr_s4", {}).get("probe_rotation_resid_deg", float("nan"))
        for arm in ("snr_s4_blur2", "snr_s4_blur4"):
            if arm in table and np.isfinite(base):
                ratio = (table[arm]["probe_rotation_resid_deg"] / base) ** 2
                pred = gmax * (s0 + v4) / (s0 + v4 + n * ratio)
                lines.append(f"| {arm} (N x {ratio:.2f} from probe) | {pred:.3f} | "
                             f"{fmt(table[arm]['G_v_mean'], table[arm]['G_v_mean_sd'])} |")
    else:
        lines.append("Not enough signal arms finished yet.")
    lines.append("")

    lines += ["## Retention arms (paired by seed, difference vs A)", "",
              "| arm | G_v mean | Δ G_v vs A | nominal MAE | Δ MAE vs A |", "| --- | --- | --- | --- | --- |"]
    ref = arms.get("ret_A", {})
    for arm in ("ret_old", "ret_A", "ret_B", "ret_C", "ret_D", "ret_E", "ret_F", "ret_H", "ret_H2"):
        if arm not in arms:
            continue
        seeds = sorted(s for s in arms[arm] if s in ref)
        dg = [arms[arm][s]["G_v_mean"] - ref[s]["G_v_mean"] for s in seeds]
        dm = [arms[arm][s]["nominal_mae_denorm"] - ref[s]["nominal_mae_denorm"] for s in seeds]
        r = table[arm]
        lines.append(f"| {arm} | {fmt(r['G_v_mean'], r['G_v_mean_sd'])} | {fmt(*mean_sd(dg))} | "
                     f"{fmt(r['nominal_mae_denorm'], r['nominal_mae_denorm_sd'], 4)} | {fmt(*mean_sd(dm), 4)} |")
    lines += ["", "Predictions: SNR law → A ≈ C, B < A, H ≈ H2 < A.  "
              "Fluent-data interference (ChatGPT H1) → C < A and H < H2.", ""]

    probe_rows = []
    for path in sorted((OUT / "probes").glob("*/probe.json")):
        p = json.loads(path.read_text())
        probe_rows.append((path.parent.name, p))
    lines += ["## GR00T probes", "",
              "| checkpoint | steps | crop | G_v row 31 [95% CI] | G_v row 0 | G_p row 31 | turn slope | yaw MAE row 31 (deg) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, p in probe_rows:
        ci = p.get("G_v_31_ci95", [float("nan")] * 2)
        lines.append(f"| {name} | {p['denoising_steps']} | {p['crop_frac']} | {p['G_v_31']:.3f} "
                     f"[{ci[0]:.3f}, {ci[1]:.3f}] | {p['G_v_0']:.3f} | {p['G_p_31']:.3f} | "
                     f"{p['turn_slope_origin']:.3f} | {p['yaw_mae_row31_deg']:.2f} |")
    lines.append("")
    calib = OUT.parent / "cf_cache" / "calibration.json"
    if calib.is_file():
        lines += ["## Warp calibration", "", "```", calib.read_text().strip(), "```", ""]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
