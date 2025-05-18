#!/usr/bin/env python3
"""
MRI GRE 6-echo 시각화 스크립트
────────────────────────────────────────────────────────────
• 입력
    ├ meas_gre_dir1.mat            ← clean  complex128  (meas_gre)
    ├ noisy_meas_gre_dir1.mat      ← noisy_real / noisy_imag
    └ denoised_real_imag.mat       ← den_real  / den_imag
• 처리
    1. magnitude 계산
    2. 6 echo → 3-D 볼륨으로 **합성**   (mean -- 기본 / RSS 주석 해제하면 사용)
    3. 지정 슬라이스(z_idx)에서  ──  Original | Noisy | Denoised  ──  3컷 PNG
• 사용
    python3 visualize_gre_combined.py --z 88 \
        --out gre_combined_z88.png
────────────────────────────────────────────────────────────
pip install numpy scipy matplotlib
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ─── CLI ──────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Combine 6 echoes and save slice PNG")
ap.add_argument("--orig", default="meas_gre_dir1.mat")
ap.add_argument("--noisy", default="noisy_meas_gre_dir1.mat")
ap.add_argument("--den", default="denoised_real_imag_10.mat")
ap.add_argument("--z", type=int, default=88, help="axial slice index (0-based)")
ap.add_argument("--out", default="gre_combined.png", help="output PNG path")
args = ap.parse_args()

# ─── load data ────────────────────────────────────────────────────────────
orig = sio.loadmat(args.orig, simplify_cells=True)
nois = sio.loadmat(args.noisy, simplify_cells=True)
den  = sio.loadmat(args.den,  simplify_cells=True)

mask = orig["mask_brain"].astype(bool)                  # (X,Y,Z)
orig_cplx = orig["meas_gre"].astype(np.complex64)       # (X,Y,Z,6)
noisy_real = nois["noisy_real"].astype(np.float32)
noisy_imag = nois["noisy_imag"].astype(np.float32)
den_real   = den["den_real"].astype(np.float32)
den_imag   = den["den_imag"].astype(np.float32)

# ─── magnitude + echo combine (mean) ──────────────────────────────────────
mag_orig  = np.abs(orig_cplx).mean(axis=3)
mag_noisy = np.abs(noisy_real + 1j*noisy_imag).mean(axis=3)
mag_den   = np.abs(den_real   + 1j*den_imag  ).mean(axis=3)

# ── RSS 방법 사용하려면 위 3줄 대신 아래 3줄 주석 해제 ──────────────────
# mag_orig  = np.sqrt((np.abs(orig_cplx)**2).sum(axis=3))
# mag_noisy = np.sqrt((np.abs(noisy_real+1j*noisy_imag)**2).sum(axis=3))
# mag_den   = np.sqrt((np.abs(den_real  +1j*den_imag )**2).sum(axis=3))

# ─── robust display range ────────────────────────────────────────────────
vmin, vmax = np.percentile(mag_orig[mask], (1, 99))

# ─── save 1×3 grid ───────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
titles = ["Original (mean 6 echo)", "Noisy", "Denoised"]
for a, t, vol in zip(ax, titles, [mag_orig, mag_noisy, mag_den]):
    a.imshow(vol[:, :, args.z], cmap="gray", vmin=vmin, vmax=vmax)
    a.set_title(t, fontsize=11)
    a.axis("off")
plt.tight_layout(pad=0.3)
plt.savefig(args.out, dpi=300, bbox_inches="tight")
plt.close()
print(f"✔ slice {args.z} grid saved → {Path(args.out).resolve()}")
