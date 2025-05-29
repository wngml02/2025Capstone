from pathlib import Path

import matplotlib.pyplot as plt
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

# 데이터 준비
mask = orig['mask_brain'].astype(bool)
orig_cplx = orig['meas_gre'].astype(np.complex64)
noisy_real = noisy['noisy_real'].astype(np.float32)
noisy_imag = noisy['noisy_imag'].astype(np.float32)
den_cplx = den['den_real'].astype(np.float32) + 1j * den['den_imag'].astype(np.float32)

# 복소수 및 magnitude 변환
noisy_cplx = noisy_real + 1j * noisy_imag
ref = np.abs(orig_cplx)
noisy = np.abs(noisy_cplx)
den = np.abs(den_cplx)

# 🌟 sigma_est 계산 (real/imag 분리 방식)
noise_real_all = noisy_real - orig_cplx.real
noise_imag_all = noisy_imag - orig_cplx.imag
noise_std_real = np.std(noise_real_all[mask])
noise_std_imag = np.std(noise_imag_all[mask])
sigma_est = np.sqrt((noise_std_real**2 + noise_std_imag**2) / 2)
print(f"🌟 Global Sigma (mean): {sigma_est:.4f}")

# Ground Truth 표준편차 계산 (real/imag 기반)
gt_std_list = []
for echo in range(orig_cplx.shape[3]):
    ref_real = orig_cplx[..., echo].real
    ref_imag = orig_cplx[..., echo].imag
    noise_std_real = np.std(ref_real[mask])
    noise_std_imag = np.std(ref_imag[mask])
    gt_std = np.sqrt((noise_std_real**2 + noise_std_imag**2) / 2)
    gt_std_list.append(gt_std)

# MSE/RMSE 계산 함수 (echo별)
def calc_mse_rmse_echo(ref, comp, mask, sigma):
    n_echoes = ref.shape[3]
    mse_list, rmse_list = [], []
    for echo in range(n_echoes):
        r = ref[..., echo]
        c = comp[..., echo]
        msk = mask
        
        # 보정 전
        mse_no = np.mean(np.square((r - c)[msk]))
        rmse_no = np.sqrt(mse_no)

        # 보정 후
        c_corr = np.sqrt(np.maximum(c**2 - 2 * sigma**2, 0))
        mse_corr = np.mean(np.square((r - c_corr)[msk]))
        rmse_corr = np.sqrt(mse_corr)

        mse_list.append((mse_no, mse_corr))
        rmse_list.append((rmse_no, rmse_corr))
    return mse_list, rmse_list

# ─── 계산 ─────────────────────────────────────
print("⋯ Noisy 계산")
mse_vals, rmse_vals = calc_mse_rmse_echo(ref, noisy, mask, sigma_est)
print("⋯ Denoised 계산")
mse_vals_den, rmse_vals_den = calc_mse_rmse_echo(ref, den, mask, sigma_est)

# ─── 표 정리 ─────────────────────────────
n_echoes = ref.shape[3]
df_mse = pd.DataFrame({
    'Echo': range(1, n_echoes+1),
    'GT_Std': gt_std_list,
    'Before MSE': [m[0] for m in mse_vals],
    'After MSE': [m[1] for m in mse_vals_den]
})
df_rmse = pd.DataFrame({
    'Echo': range(1, n_echoes+1),
    'GT_Std': gt_std_list,
    'Before RMSE': [r[0] for r in rmse_vals],
    'After RMSE': [r[1] for r in rmse_vals_den]
})

# ─── RMSE 감소 비율(%) 계산 및 추가 ─────────────────────────────
df_rmse['RMSE Reduction (%)'] = 100 * (df_rmse['Before RMSE'] - df_rmse['After RMSE']) / df_rmse['Before RMSE']

# ─── 출력 및 저장 ─────────────────────────────
output_path = Path("MSE_RMSE_Mag_BiasCorrected_GTstd.xlsx")
with pd.ExcelWriter(output_path) as writer:
    df_mse.to_excel(writer, sheet_name="MSE_Table", index=False)
    df_rmse.to_excel(writer, sheet_name="RMSE_Table", index=False)
print(f"✔ Excel 저장 완료: {output_path}")

print("\n=== MSE Table (GT Std 포함) ===")
print(df_mse.round(4))
print("\n=== RMSE Table (GT Std 포함 + 감소율) ===")
print(df_rmse.round(4))

# ─── RMSE Bar Chart ─────────────────────────────
plt.figure(figsize=(10, 6))
x = df_rmse['Echo']
plt.bar(x - 0.15, df_rmse['Before RMSE'], width=0.3, label='Before RMSE')
plt.bar(x + 0.15, df_rmse['After RMSE'], width=0.3, label='After RMSE')
plt.xlabel('Echo')
plt.ylabel('RMSE')
plt.title('Before vs After RMSE by Echo')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
