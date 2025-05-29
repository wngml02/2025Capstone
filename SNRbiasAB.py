import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# 파일 로드
ORIG_MAT = "meas_gre_dir1.mat"
DENO_MAT = "denoised_real_imag_30_sqrt_r3.mat"
NOISY_MAT = "noisy_meas_gre_dir1_30.mat"

print("⋯ 데이터 로드")
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
den = sio.loadmat(DENO_MAT, simplify_cells=True)
noisy = sio.loadmat(NOISY_MAT, simplify_cells=True)

mask = orig['mask_brain'].astype(bool)
orig_cplx = orig['meas_gre'].astype(np.complex64)
den_real = den['den_real'].astype(np.float32)
den_imag = den['den_imag'].astype(np.float32)
noisy_real = noisy['noisy_real'].astype(np.float32)
noisy_imag = noisy['noisy_imag'].astype(np.float32)

# magnitude 생성
den_cplx = den_real + 1j * den_imag
mag_den = np.abs(den_cplx)
mag_orig = np.abs(orig_cplx)
noisy_cplx = noisy_real + 1j * noisy_imag
mag_noisy = np.abs(noisy_cplx)

noise_real_all = noisy_real - orig_cplx.real
noise_imag_all = noisy_imag - orig_cplx.imag
noise_std_real_all = np.std(noise_real_all[mask])
noise_std_imag_all = np.std(noise_imag_all[mask])
sigma_scalar = np.sqrt((noise_std_real_all**2 + noise_std_imag_all**2) / 2)
print(f"Global Sigma (mean): {sigma_scalar:.4f}")

# SNR 계산 함수
def compute_snr(data, ref, mask, is_complex=True):
    if is_complex:
        # 복소수 데이터 → magnitude 파워 기반 SNR
        signal_power = np.mean(np.abs(ref[mask])**2)
        noise_power = np.mean(np.abs(data[mask] - ref[mask])**2)
    else:
        # 실수 데이터 → 그냥 제곱 기반 SNR
        signal_power = np.mean((ref[mask])**2)
        noise_power = np.mean((data[mask] - ref[mask])**2)
    return 10 * np.log10(signal_power / noise_power)
# Echo별 보정 전후 비교
n_echoes = orig_cplx.shape[-1]
rows = []
for e in range(n_echoes):
    m_o = mag_orig[..., e]
    m_n = mag_noisy[..., e]
    m_d = mag_den[..., e]
    
    m_nc = np.sqrt(np.maximum(m_n**2 - 2*sigma_scalar**2, 0))
    m_dc = np.sqrt(np.maximum(m_d**2 - 2*sigma_scalar**2, 0))
    # SNR/SSIM 계산 (ground truth vs denoised)
    snr_b = compute_snr(m_d, m_o, mask)      # 보정 전
    snr_a = compute_snr(m_dc, m_o, mask)     # 보정 후
    ssim_b = ssim(m_o, m_d, data_range=np.ptp(m_o), mask=mask)
    ssim_a = ssim(m_o, m_dc, data_range=np.ptp(m_o), mask=mask)
    
    rows.append(dict(Echo=e, SNR_before=snr_b, SNR_after=snr_a, ΔSNR=snr_a-snr_b,
                    SSIM_before=ssim_b, SSIM_after=ssim_a, ΔSSIM=ssim_a-ssim_b))

# DataFrame 생성
df = pd.DataFrame(rows)
average = df.mean(numeric_only=True)
average_row = {col: average[col] for col in df.columns if col != 'Echo'}
average_row['Echo'] = 'Average'
df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

# 출력
print("\nEcho-wise Performance Comparison (Denoised: Bias Correction Before vs After, Ground Truth 기준):")
print(df.round(4))

# 필요하면 Excel 저장
with pd.ExcelWriter("Denoised_BiasCorrection_Comparison.xlsx") as writer:
    df.to_excel(writer, sheet_name="Denoised_BiasCorrection", index=False)
print("✔ Excel 저장 완료: Denoised_BiasCorrection_Comparison.xlsx")
