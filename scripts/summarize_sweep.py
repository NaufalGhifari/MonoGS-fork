#!/usr/bin/env python3
"""Summarise a gaussian_th sweep from each run's results.json + config.yml.

Each run directory written by slam.py contains both files, so gaussian_th and
the dataset protocol come from config.yml while the metrics come from
results.json (which does not record the swept parameter).

Examples:
    python scripts/summarize_sweep.py
    python scripts/summarize_sweep.py --root results/sweeps/gth --csv sweep.csv
"""

import argparse
import csv
import glob
import json
import os
import sys

import yaml

COLUMNS = (
    ("th", 6, "{:<6}"),
    ("gaussians", 10, "{:>10}"),
    ("KFs", 5, "{:>5}"),
    ("ATE", 8, "{:>8}"),
    ("PSNR_b", 8, "{:>8}"),
    ("PSNR_a", 8, "{:>8}"),
    ("SSIM_a", 8, "{:>8}"),
    ("LPIPS_a", 8, "{:>8}"),
    ("GPU%", 6, "{:>6}"),
    ("total_min", 10, "{:>10}"),
    ("N/N0", 7, "{:>7}"),
    ("dPSNR", 8, "{:>8}"),
    ("T/T0", 7, "{:>7}"),
)


def dig(data, *keys, default=None):
    """Walk nested dicts, returning ``default`` on any missing/non-dict hop."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def fmt(value, spec="{:>8}", na="n/a"):
    if value is None:
        return na
    try:
        return spec.format(value)
    except (TypeError, ValueError):
        return str(value)


def scene_name(dataset_path):
    if not dataset_path:
        return "unknown"
    return str(dataset_path).rstrip("/").split("/")[-1]


def load_config(run_dir):
    path = os.path.join(run_dir, "config.yml")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def load_results(run_dir):
    path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def build_row(run_dir):
    """Return a metrics row, or None if the run has no result payload."""
    results = load_results(run_dir)
    if results is None:
        return None

    config = load_config(run_dir) or {}
    dataset = config.get("Dataset", {}) if isinstance(config, dict) else {}
    training = config.get("Training", {}) if isinstance(config, dict) else {}

    rendering = dig(results, "metrics", "rendering", default={}) or {}
    before = rendering.get("before_opt") or {}
    after = rendering.get("after_opt") or {}

    gpu_fraction = dig(results, "compute", "memory", "peak_gpu_used_fraction")
    total_s = dig(results, "compute", "timing", "total_s")

    return {
        "run_dir": run_dir,
        "scene": scene_name(dataset.get("dataset_path")),
        "dataset_path": dataset.get("dataset_path"),
        "config_path": results.get("run", {}).get("config_path"),
        "gaussian_th": training.get("gaussian_th"),
        "stride": dataset.get("stride", 1),
        "num_frames": dig(results, "dataset", "num_frames"),
        "num_keyframes": dig(results, "compute", "model", "num_keyframes"),
        "num_gaussians": dig(results, "compute", "model", "num_gaussians_final"),
        "ate_rmse": dig(results, "metrics", "ate", "rmse"),
        "psnr_before": before.get("mean_psnr"),
        "psnr_after": after.get("mean_psnr"),
        "ssim_after": after.get("mean_ssim"),
        "lpips_after": after.get("mean_lpips"),
        "gpu_fraction": gpu_fraction,
        "total_s": total_s,
    }


def collect(root):
    """Return (completed rows, failed run directories) under ``root``."""
    rows = [row for row in (build_row(os.path.dirname(p)) for p in
                            glob.glob(os.path.join(root, "**", "results.json"),
                                      recursive=True))
            if row is not None]

    failed = []
    for cfg_path in glob.glob(os.path.join(root, "**", "config.yml"),
                              recursive=True):
        run_dir = os.path.dirname(cfg_path)
        if not os.path.isfile(os.path.join(run_dir, "results.json")):
            failed.append(run_dir)

    return rows, sorted(set(failed))


def pick_baseline(rows, requested):
    """Baseline row used for normalised columns (densest map by default)."""
    candidates = [r for r in rows if r["gaussian_th"] is not None]
    if not candidates:
        return None
    if requested is not None:
        for row in candidates:
            if abs(float(row["gaussian_th"]) - requested) < 1e-9:
                return row
        return None
    return min(candidates, key=lambda r: float(r["gaussian_th"]))


def normalise(rows, baseline):
    for row in rows:
        for key in ("gauss_ratio", "d_psnr", "time_ratio"):
            row[key] = None
        if baseline is None:
            continue
        if row["num_gaussians"] and baseline["num_gaussians"]:
            row["gauss_ratio"] = row["num_gaussians"] / baseline["num_gaussians"]
        if row["psnr_after"] is not None and baseline["psnr_after"] is not None:
            row["d_psnr"] = row["psnr_after"] - baseline["psnr_after"]
        if row["total_s"] and baseline["total_s"]:
            row["time_ratio"] = row["total_s"] / baseline["total_s"]


def render_scene(scene, rows):
    protocol = rows[0]
    stride = protocol.get("stride") or 1
    header = "".join(spec.format(name) for name, _, spec in COLUMNS)
    print(f"\nscene: {scene}   frames={fmt(protocol['num_frames'], '{}')} "
          f" stride={stride}")
    print(header)
    print("-" * len(header))
    for row in rows:
        values = (
            fmt(row["gaussian_th"], "{:<6.3f}"),
            fmt(row["num_gaussians"], "{:>10,}"),
            fmt(row["num_keyframes"], "{:>5}"),
            fmt(row["ate_rmse"], "{:>8.4f}"),
            fmt(row["psnr_before"], "{:>8.2f}"),
            fmt(row["psnr_after"], "{:>8.2f}"),
            fmt(row["ssim_after"], "{:>8.4f}"),
            fmt(row["lpips_after"], "{:>8.4f}"),
            fmt(row["gpu_fraction"] * 100 if row["gpu_fraction"] is not None
                else None, "{:>5.1f}%"),
            fmt(row["total_s"] / 60 if row["total_s"] else None, "{:>10.1f}"),
            fmt(row["gauss_ratio"], "{:>7.3f}"),
            fmt(row["d_psnr"], "{:>+8.3f}"),
            fmt(row["time_ratio"], "{:>7.3f}"),
        )
        print("".join(values))


def write_csv(path, rows):
    fields = ("scene", "gaussian_th", "stride", "num_frames", "num_keyframes",
              "num_gaussians", "gauss_ratio", "ate_rmse", "psnr_before",
              "psnr_after", "d_psnr", "ssim_after", "lpips_after",
              "gpu_fraction", "total_s", "time_ratio", "run_dir")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarise gaussian_th sweep runs found under a directory"
    )
    parser.add_argument("--root", default="results/sweeps",
                        help="directory searched recursively (default: results/sweeps)")
    parser.add_argument("--baseline", type=float, default=None,
                        help="gaussian_th used for normalised columns "
                             "(default: lowest threshold found per scene)")
    parser.add_argument("--csv", default=None,
                        help="write all rows to this CSV path")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"No such directory: {args.root}")

    rows, failed = collect(args.root)
    if not rows:
        sys.exit(f"No runs with results.json found under {args.root}")

    rows.sort(key=lambda r: (r["scene"], r["gaussian_th"] is None, r["gaussian_th"]))

    scenes = {}
    for row in rows:
        scenes.setdefault(row["scene"], []).append(row)

    for scene in sorted(scenes):
        scene_rows = sorted(
            scenes[scene],
            key=lambda r: (r["gaussian_th"] is None, r["gaussian_th"]),
        )
        normalise(scene_rows, pick_baseline(scene_rows, args.baseline))
        render_scene(scene, scene_rows)

    if failed:
        print(f"\n{len(failed)} run(s) without results.json "
              f"(crashed or still running):")
        for run_dir in failed:
            print(f"  {run_dir}")

    if args.csv:
        write_csv(args.csv, rows)


if __name__ == "__main__":
    main()
