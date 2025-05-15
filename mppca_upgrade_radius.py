#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import localpca, mppca

# ── 공통 입력 ───────────────────────────────────────────────────────────
ORIG_MAT  = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1.mat"
Z_SLICE   = 88
PATCH_R_LIST = [1, 2, 3]          # ← 시험할 radius 값들

# ── 데이터 1회만 로드 ───────────────────────────────────────────────────
m0 = sio.loadmat(ORIG_MAT,  simplify_cells=True)
orig_cplx = m0["meas_gre"].astype(np.complex64)
mask      = m0["mask_brain"].astype(bool)

m1 = sio.loadmat(NOISY_MAT, simplify_cells=True)
noisy_real = m1["noisy_real"].astype(np.float32)
noisy_imag = m1["noisy_imag"].astype(np.float32)

# ── MP-PCA 함수 (부호 보존) ─────────────────────────────────────────────
def mppca_denoise(vol4d, r):
    sign, absvol = np.sign(vol4d), np.abs(vol4d)
    sigma4d = mppca(absvol, mask=mask, patch_radius=r)
    sigma3d = sigma4d.mean(axis=3).astype(np.float32)
    den_abs = localpca(absvol, sigma=sigma3d, mask=mask,
                       patch_radius=r).astype(np.float32)
    return den_abs * sign

# ── 반경별 루프 ────────────────────────────────────────────────────────
for R in PATCH_R_LIST:
    print(f"\n=== PATCH_RADIUS = {R} ===")

    # 1) 디노이즈
    den_real = mppca_denoise(noisy_real, R)
    den_imag = mppca_denoise(noisy_imag, R)
    sio.savemat(f"denoised_real_imag_r{R}.mat",
                {"den_real": den_real, "den_imag": den_imag})
    print(f"✔ .mat 저장 → denoised_real_imag_r{R}.mat")

    # 2) 그리드 PNG
    mag_orig  = np.abs(orig_cplx)
    mag_noisy = np.abs(noisy_real + 1j*noisy_imag)
    mag_den   = np.abs(den_real   + 1j*den_imag)

    vmin, vmax = np.percentile(mag_orig[mask], (1, 99))

    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    rows = [("Original", mag_orig),
            ("Noisy",    mag_noisy),
            ("Denoised", mag_den)]
    titles = [f"Echo {i+1}" for i in range(6)]

    for r, (label, vol) in enumerate(rows):
        for e in range(6):
            ax = axes[r, e]
            ax.imshow(vol[:, :, Z_SLICE, e], cmap="gray",
                      vmin=vmin, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[e], fontsize=11)
            if e == 0:
                ax.set_ylabel(label, fontsize=12, rotation=0, labelpad=40)

    plt.tight_layout(pad=0.3)
    out_png = f"gre_mp_pca_grid_r{R}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔ 그리드 저장 → {out_png}")
