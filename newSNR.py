from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dipy.denoise.noise_estimate import estimate_sigma
from skimage.metrics import structural_similarity as ssim

# 파일 로드
ORIG_MAT = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_50.mat"
DENO_MAT = "denoised_real_imag_50_sqrt_r3.mat"

print("⋯ 데이터 로드")
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
noisy = sio.loadmat(NOISY_MAT, simplify_cells=True)
den = sio.loadmat(DENO_MAT, simplify_cells=True)

mask = orig['mask_brain'].astype(bool)
orig_cplx = orig['meas_gre'].astype(np.complex64)
noisy_real = noisy['noisy_real'].astype(np.float32)
noisy_imag = noisy['noisy_imag'].astype(np.float32)
den_real = den['den_real'].astype(np.float32)
den_imag = den['den_imag'].astype(np.float32)

# 복소수 및 magnitude 생성
noisy_cplx = noisy_real + 1j * noisy_imag
den_cplx = den_real + 1j * den_imag
mag_noisy = np.abs(noisy_cplx)
mag_den = np.abs(den_cplx)
mag_orig = np.abs(orig_cplx)

# Noise sigma 추정 및 correction
sigma_est = estimate_sigma(mag_noisy[mask].reshape(-1,1,1,1), N=4)
sigma_scalar = float(np.mean(sigma_est))
print(f"Estimated noise sigma: {sigma_scalar:.4f}")
mag_noisy_corr = np.sqrt(np.maximum(mag_noisy**2 - 2*sigma_scalar**2, 0))
mag_den_corr = np.sqrt(np.maximum(mag_den**2 - 2*sigma_scalar**2, 0))

# SNR 계산 함수
def compute_snr(data, ref, mask, is_complex=True):
    if is_complex:
        mean_signal = np.abs(np.mean(ref[mask]))
        noise_std = np.std(data[mask] - ref[mask])
    else:
        mean_signal = np.mean(ref[mask])
        noise_std = np.std(data[mask] - ref[mask])
    return 20 * np.log10(mean_signal / noise_std)

# Echo별 결과 분리
n_echoes = orig_cplx.shape[-1]
rows_complex, rows_mag, rows_corr = [], [], []

for e in range(n_echoes):
    o = orig_cplx[..., e]
    n_c = noisy_cplx[..., e]
    d_c = den_cplx[..., e]
    m_o = mag_orig[..., e]
    m_n = mag_noisy[..., e]
    m_d = mag_den[..., e]
    m_nc = mag_noisy_corr[..., e]
    m_dc = mag_den_corr[..., e]
    
    # 복소수 SNR
    snr_b_c = compute_snr(n_c, o, mask, is_complex=True)
    snr_a_c = compute_snr(d_c, o, mask, is_complex=True)
    rows_complex.append(dict(SNR_before=snr_b_c, SNR_after=snr_a_c, ΔSNR=snr_a_c - snr_b_c))
    
    # magnitude SNR
    snr_b_m = compute_snr(m_n, m_o, mask, is_complex=False)
    snr_a_m = compute_snr(m_d, m_o, mask, is_complex=False)
    rows_mag.append(dict(SNR_before=snr_b_m, SNR_after=snr_a_m, ΔSNR=snr_a_m - snr_b_m))
    
    # magnitude correction SNR
    snr_b_corr = compute_snr(m_nc, m_o, mask, is_complex=False)
    snr_a_corr = compute_snr(m_dc, m_o, mask, is_complex=False)
    rows_corr.append(dict(SNR_before=snr_b_corr, SNR_after=snr_a_corr, ΔSNR=snr_a_corr - snr_b_corr))

# DataFrames 생성
df_complex = pd.DataFrame(rows_complex)
df_magnitude = pd.DataFrame(rows_mag)
df_corrected = pd.DataFrame(rows_corr)

# 출력
print("\n=== Complex SNR by Echo ===")
print(df_complex.round(3))
print("\n=== Magnitude SNR by Echo ===")
print(df_magnitude.round(3))
print("\n=== Corrected Magnitude SNR by Echo ===")
print(df_corrected.round(3))

# 필요하면 Excel 저장
with pd.ExcelWriter("SNR_by_Echo.xlsx") as writer:
    df_complex.to_excel(writer, sheet_name="Complex_SNR", index=False)
    df_magnitude.to_excel(writer, sheet_name="Magnitude_SNR", index=False)
    df_corrected.to_excel(writer, sheet_name="Corrected_SNR", index=False)
print("✔ Excel 저장 완료: SNR_by_Echo.xlsx")


# Echo 개수 (x축)
import matplotlib.pyplot as plt

# Echo 개수 (x축)
x = df_magnitude.index

plt.figure(figsize=(10, 5))
plt.plot(x, df_magnitude['SNR_before'], marker='o', label='Original Magnitude SNR Before')
plt.plot(x, df_magnitude['SNR_after'], marker='o', label='Original Magnitude SNR After')
plt.plot(x, df_corrected['SNR_before'], marker='s', label='Corrected Magnitude SNR Before')
plt.plot(x, df_corrected['SNR_after'], marker='s', label='Corrected Magnitude SNR After')
plt.title('Magnitude SNR Comparison: Original vs Corrected')
plt.xlabel('Echo')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)
plt.show()