#!/usr/bin/env python3
"""Simulate low-light capture on a TUM-style rgb/ folder.

Pipeline per frame (done in linear light, not on gamma-encoded pixels):
  sRGB -> linear -> scale by 2^-stops (less light) -> Poisson shot noise
  -> Gaussian read noise -> (optional) re-amplify to the original brightness
  -> sRGB -> 8-bit.

Filenames are preserved, so associations.txt / rgb.txt / groundtruth.txt stay valid.
By default a full dataset root is created next to the original, with the darkened
rgb/ folder and symlinks to everything else (depth/, associations.txt, ...).

Examples:
  python darken.py tum_dataset/run1/rgb --stops 1 2 3
  python darken.py tum_dataset/run1/rgb --stops 3 --normalize     # dark + high-ISO noise look
"""
import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

EXTS = {".png", ".jpg", ".jpeg"}


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def darken(img_bgr, stops, peak, read, normalize, rng):
    x = srgb_to_linear(img_bgr.astype(np.float64) / 255.0)
    gain = 2.0 ** (-stops)
    # Shot noise: photon count at full white is `peak`, so darker pixels get fewer
    # photons and a worse signal-to-noise ratio.
    y = rng.poisson(x * gain * peak).astype(np.float64) / peak
    y += rng.normal(0.0, read, size=y.shape)          # sensor read noise (linear units)
    if normalize:                                      # auto-gain: restore brightness, keep noise
        y /= gain
    out = linear_to_srgb(y)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def link_siblings(src_root, dst_root, skip_name):
    for item in src_root.iterdir():
        if item.name == skip_name:
            continue
        dst = dst_root / item.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(item.resolve(), dst)
        except OSError:                                # e.g. symlinks not permitted
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rgb_dir", help="folder with the RGB frames, e.g. tum_dataset/run1/rgb")
    ap.add_argument("--stops", type=float, nargs="+", default=[2.0],
                    help="how many stops (EV) darker; 1 = half the light. One dataset per value")
    ap.add_argument("--peak", type=float, default=1000.0,
                    help="photons at full white; lower = noisier (default 1000)")
    ap.add_argument("--read", type=float, default=0.005,
                    help="read-noise std in linear units, 0-1 scale (default 0.005)")
    ap.add_argument("--normalize", action="store_true",
                    help="re-brighten after adding noise (mimics auto-gain / high ISO)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dataset root (single --stops value only)")
    ap.add_argument("--no-link", action="store_true", help="don't link depth/, associations.txt, etc.")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir).resolve()
    root = rgb_dir.parent
    files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images found in {rgb_dir}")
    if args.out and len(args.stops) > 1:
        raise SystemExit("--out can only be used with a single --stops value")

    for stops in args.stops:
        tag = f"dark{stops:g}ev" + ("_norm" if args.normalize else "")
        dst_root = Path(args.out) if args.out else root.parent / f"{root.name}_{tag}"
        out_rgb = dst_root / rgb_dir.name
        out_rgb.mkdir(parents=True, exist_ok=True)

        in_mean = out_mean = 0.0
        for i, p in enumerate(files):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            rng = np.random.default_rng([args.seed, i])   # deterministic per frame
            out = darken(img, stops, args.peak, args.read, args.normalize, rng)
            cv2.imwrite(str(out_rgb / p.name), out, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            in_mean += img.mean()
            out_mean += out.mean()

        if not args.no_link:
            link_siblings(root, dst_root, rgb_dir.name)
        n = len(files)
        print(f"[{tag}] {n} frames -> {dst_root}  (mean intensity {in_mean/n:.1f} -> {out_mean/n:.1f})")


if __name__ == "__main__":
    main()