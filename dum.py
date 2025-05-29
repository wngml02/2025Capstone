from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# ─── 파일 로드 ─────────────────────────────
ORIG_MAT = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_30.mat"
DENO_MAT = "denoised_real_imag_30_sqrt_r3.mat"

print("⋯ 데이터 로드")
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
noisy = sio.loadmat(NOISY_MAT, simplify_cells=True)
den = sio.loadmat(DENO_MAT, simplify_cells=True)

# 마스크, 복소수 데이터 준비
mask = orig['mask_brain'].astype(bool)
ref = orig['meas_gre'].astype(np.complex64)
noisy = noisy['noisy_real'].astype(np.float32) + 1j * noisy['noisy_imag'].astype(np.float32)
den = den['den_real'].astype(np.float32) + 1j * den['den_imag'].astype(np.float32)

# 노이즈 표준편차 추정
sigma_est = np.std((noisy - ref)[mask])
print(f"Estimated noise sigma: {sigma_est:.4f}")

# MSE/RMSE 계산 함수
def calc_mse_rmse(ref, comp, mask, sigma):
    n_echoes, n_slices = ref.shape[3], ref.shape[2]
    results = []
    for echo in range(n_echoes):
        for slice_idx in range(n_slices):
            msk = mask[:, :, slice_idx]
            r = ref[:, :, slice_idx, echo]
            c = comp[:, :, slice_idx, echo]

            # Ground Truth vs Ground Truth (baseline)
            mse_gt, rmse_gt = 0, 0

            # 보정 전
            diff_no = (r - c)[msk]
            mse_no = np.mean(np.square(diff_no))
            rmse_no = np.sqrt(mse_no)

            # 보정 후 (magnitude + bias correction)
            c_mag_corr = np.sqrt(np.maximum(np.abs(c)**2 - 2 * sigma**2, 0))
            r_mag_corr = np.sqrt(np.maximum(np.abs(r)**2 - 2 * sigma**2, 0))
            diff_corr = (r_mag_corr - c_mag_corr)[msk]
            mse_corr = np.mean(np.square(diff_corr))
            rmse_corr = np.sqrt(mse_corr)

            results.append({
                'Echo': echo+1, 'Slice': slice_idx+1,
                'MSE_GT': mse_gt, 'RMSE_GT': rmse_gt,
                'MSE_before': mse_no, 'RMSE_before': rmse_no,
                'MSE_after': mse_corr, 'RMSE_after': rmse_corr
            })
    return pd.DataFrame(results)

# ─── 계산 ─────────────────────────────────────
print("⋯ Noisy vs Ground Truth 계산")
df_noisy = calc_mse_rmse(ref, noisy, mask, sigma_est)

print("⋯ Denoised vs Ground Truth 계산")
df_den = calc_mse_rmse(ref, den, mask, sigma_est)

# ─── 출력 및 저장 ─────────────────────────────
print("\n=== Noisy vs Ground Truth ===")
print(df_noisy.round(4))

print("\n=== Denoised vs Ground Truth ===")
print(df_den.round(4))

output_path = Path("MSE_RMSE_Comparison.xlsx")
with pd.ExcelWriter(output_path) as writer:
    df_noisy.to_excel(writer, sheet_name="Noisy_vs_GT", index=False)
    df_den.to_excel(writer, sheet_name="Denoised_vs_GT", index=False)
print(f"✔ Excel 저장 완료: {output_path}")
