from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from localpca_dn import mppca

# ─── 사용자 설정 ────────────────────────────────────────────────────────────
ORIG_MAT  = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_10.mat"
OUT_MAT   = "denoised_real_imag_10_dn2.mat"
GRID_PNG  = "gre_mp_pca_grid_dn2.png"

PATCH_R   = 1
Z_SLICE   = 88

# ─── 데이터 로드 ────────────────────────────────────────────────────────────
print("⋯ 원본 GRE + 마스크 로드")
m0            = sio.loadmat(ORIG_MAT, simplify_cells=True)
orig_cplx     = m0["meas_gre"].astype(np.complex64)         # (256,224,176,6)
mask          = m0["mask_brain"].astype(bool)

print("⋯ 노이즈 real / imag 로드")
m1           = sio.loadmat(NOISY_MAT, simplify_cells=True)
noisy_real   = m1["noisy_real"].astype(np.float32)
noisy_imag   = m1["noisy_imag"].astype(np.float32)
assert noisy_real.shape == orig_cplx.shape, "shape mismatch!"

# ─── MP-PCA 디노이즈 함수 (부호 보존) ───────────────────────────────────────
def mppca_denoise(vol4d: np.ndarray, *, mask: np.ndarray,  patch_r: int = PATCH_R) -> np.ndarray:
    return mppca(vol4d, mask=mask, patch_radius=patch_r)

print("⋯ MP-PCA denoise REAL")
den_real = mppca_denoise(noisy_real, mask=mask)

print("⋯ MP-PCA denoise IMAG")
den_imag = mppca_denoise(noisy_imag, mask=mask)

sio.savemat(OUT_MAT, {"den_real": den_real, "den_imag": den_imag})
print(f"✔ 디노이즈 결과 저장 → {OUT_MAT}")

# ─── magnitude 계산 & 고정 display range ──────────────────────────────────
mag_orig  = np.abs(orig_cplx)
mag_noisy = np.abs(noisy_real + 1j*noisy_imag)
mag_den   = np.abs(den_real   + 1j*den_imag)

vmin, vmax = np.percentile(mag_orig[mask], (1, 99))

# ─── 3 × 6 그리드 (slice 88) 단일 PNG ───────────────────────────────────────
print("⋯ 그리드 이미지 생성")
fig, axes = plt.subplots(3, 6, figsize=(16, 8))
row_tuples = [("Original", mag_orig),
            ("Noisy",    mag_noisy),
            ("Denoised", mag_den)]
echo_titles = [f"Echo {i+1}" for i in range(6)]

for r, (row_name, vol) in enumerate(row_tuples):
    for e in range(6):
        ax = axes[r, e]
        ax.imshow(vol[:, :, Z_SLICE, e], cmap="gray", vmin=vmin, vmax=vmax)
        ax.axis("off")
        if r == 0:
            ax.set_title(echo_titles[e], fontsize=11)
        if e == 0:
            ax.set_ylabel(row_name, fontsize=12, rotation=0, labelpad=40)

plt.tight_layout(pad=0.3)
plt.savefig(GRID_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"✔ 그리드 저장 → {GRID_PNG}")