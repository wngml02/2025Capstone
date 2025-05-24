from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from localpca_dn import mppca

# ─── 사용자 설정 ────────────────────────────────────────────────────────────
ORIG_MAT  = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_30.mat"
OUT_MAT   = "denoised_real_imag_30_sqrt_r3.mat"

OUT_DIR = Path("dn_30_rd_3")     # 원하는 폴더 이름
OUT_DIR.mkdir(parents=True, exist_ok=True)
PATCH_R   = 2
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
mag_orig  = np.sqrt(np.square(orig_cplx.real) + np.square(orig_cplx.imag))
mag_noisy = np.sqrt(np.square(noisy_real) + np.square(noisy_imag))
mag_den   = np.sqrt(np.square(den_real) + np.square(den_imag))

vmin, vmax = np.percentile(mag_orig[mask], (1, 99))

# ─── 3 × 6 그리드 (slice 88) 단일 PNG ───────────────────────────────────────
# ─── ∆  시각화 (5-열 그리드) ───────────────────────────────────────────────
print("⋯ echo-wise 5-column 시각화 생성")
n_echoes = mag_orig.shape[-1]
slice_idx = Z_SLICE                 # 한 슬라이스 기준
diff_vmax = np.percentile(
    np.abs(mag_noisy - mag_orig)[mask], 99)  # diff 컬러범위 고정

for echo in range(n_echoes):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"Echo {echo+1}  |  Slice {slice_idx}", fontsize=16)

    # 0) Noisy magnitude
    axes[0].imshow(mag_noisy[:, :, slice_idx, echo],
                cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Magnitude')
    axes[0].axis('off')

    # 1) Denoised magnitude
    axes[1].imshow(mag_den[:, :, slice_idx, echo],
                cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised Magnitude')
    axes[1].axis('off')

    # 2) Mask
    axes[2].imshow(mask[:, :, slice_idx], cmap='gray')
    axes[2].set_title('Mask Region')
    axes[2].axis('off')

    # 3) Diff (Denoised – Original)
    diff_map = mag_den[:, :, :, echo] - mag_orig[:, :, :, echo]
    axes[3].imshow(diff_map[:, :, slice_idx],
                cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[3].set_title('Diff (Denoised − Original)')
    axes[3].axis('off')

    # 4) Diff (Noisy – Original)
    diff_noisy_map = mag_noisy[:, :, :, echo] - mag_orig[:, :, :, echo]
    axes[4].imshow(diff_noisy_map[:, :, slice_idx],
                cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[4].set_title('Diff (Noisy − Original)')
    axes[4].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = OUT_DIR / f"echo{echo+1:02d}_slice{slice_idx:03d}_grid.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✔ 저장 → {out_png}")