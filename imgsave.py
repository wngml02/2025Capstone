from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ─── 사용자 설정 ────────────────────────────────────────────────────────────
ORIG_MAT = "meas_gre_dir1.mat"
NOISY_MAT_TEMPLATE = "noisy_meas_gre_dir1_{}.mat"           # {} → 10,20,30,40,50
DENOISED_MAT_TEMPLATE = "denoised_real_imag_{}_sqrt_r3.mat" # {} → 10,20,30,40,50
Z_SLICE = 88
NOISE_LEVELS = [10, 20, 30, 40, 50]
OUT_DIR = Path("denoised_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ECHO_IDX = 0    # 첫번째 echo 선택

# ─── 원본 데이터 로드 ──────────────────────────────────────────────────────
m0 = sio.loadmat(ORIG_MAT, simplify_cells=True)
orig_cplx = m0["meas_gre"].astype(np.complex64)
mask = m0["mask_brain"].astype(bool)
orig_mag = np.abs(orig_cplx)

# ─── 노이즈 레벨별 처리 ────────────────────────────────────────────────────
for lvl in NOISE_LEVELS:
    print(f"⋯ Processing noise level {lvl}")
    noisy_file = NOISY_MAT_TEMPLATE.format(lvl)
    deno_file = DENOISED_MAT_TEMPLATE.format(lvl)
    
    # 파일 로드
    m1 = sio.loadmat(noisy_file, simplify_cells=True)
    m2 = sio.loadmat(deno_file, simplify_cells=True)
    
    noisy_real, noisy_imag = m1["noisy_real"], m1["noisy_imag"]
    den_real, den_imag = m2["den_real"], m2["den_imag"]
    
    # magnitude 계산 (echo=0)
    mag_noisy = np.sqrt(noisy_real[..., ECHO_IDX]**2 + noisy_imag[..., ECHO_IDX]**2)
    mag_den = np.sqrt(den_real[..., ECHO_IDX]**2 + den_imag[..., ECHO_IDX]**2)
    mag_orig_sel = orig_mag[..., ECHO_IDX]   # 원본도 echo=0 선택
    
    # diff 계산
    diff_deno_orig = mag_den - mag_orig_sel
    diff_noisy_orig = mag_noisy - mag_orig_sel
    
    # 컬러 범위 설정
    vmin, vmax = np.percentile(mag_orig_sel[mask], (1, 99))
    diff_vmax = np.percentile(np.abs(diff_noisy_orig[mask]), 99)
    
    # 시각화
    fig, axes = plt.subplots(1, 5, figsize=(25, 8))
    fig.suptitle(f"Noise Level {lvl} | Slice {Z_SLICE} | Echo {ECHO_IDX}", fontsize=30)

    axes[0].imshow(mag_noisy[:, :, Z_SLICE], cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Noisy Magnitude', fontsize=22)
    axes[0].axis('off')

    axes[1].imshow(mag_den[:, :, Z_SLICE], cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title('Denoised split', fontsize=22)
    axes[1].axis('off')

    axes[2].imshow(mask[:, :, Z_SLICE], cmap='gray')
    axes[2].set_title('Mask Region', fontsize=22)
    axes[2].axis('off')

    axes[3].imshow(diff_deno_orig[:, :, Z_SLICE], cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[3].set_title('Diff (Denoised - Orig)', fontsize=22)
    axes[3].axis('off')

    axes[4].imshow(diff_noisy_orig[:, :, Z_SLICE], cmap='bwr', vmin=-diff_vmax, vmax=diff_vmax)
    axes[4].set_title('Diff (Noisy - Orig)', fontsize=22)
    axes[4].axis('off')

    # ✅ 여백 조정
    plt.subplots_adjust(wspace=0.05, hspace=0, top=0.93)  # → 간격 완전 제거
    out_png = OUT_DIR / f"noise{lvl}_slice{Z_SLICE:03d}_echo{ECHO_IDX:02d}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✔ 저장 → {out_png}")
