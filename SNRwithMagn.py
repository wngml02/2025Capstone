import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# 파일 로드
ORIG_MAT = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_30.mat"
DENO_MAT = "denoised_real_imag_30_sqrt_r3.mat"

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

# 🌟 Ground truth 기반 global sigma 계산 🌟
# noisy와 ground truth(orig_cplx) 간의 real/imag 차이를 계산
noise_real_all = noisy_real - orig_cplx.real
noise_imag_all = noisy_imag - orig_cplx.imag

# mask 영역(신호 있는 영역)에서 표준편차 계산
noise_std_real = np.std(noise_real_all[mask])
noise_std_imag = np.std(noise_imag_all[mask])

# real/imag 두 성분의 분산을 평균한 후 표준편차(sigma) 계산
sigma_global = np.sqrt((noise_std_real**2 + noise_std_imag**2) / 2)
print(f"🌟 Global Sigma (mean): {sigma_global:.4f}")

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

# Echo별 결과 리스트 생성
n_echoes = orig_cplx.shape[-1]
rows = []
for e in range(n_echoes):
    m_o = mag_orig[..., e]  # ground truth magnitude
    m_n = mag_noisy[..., e] # noisy magnitude
    m_d = mag_den[..., e]   # denoised magnitude
    
    # 🌟 Bias correction 공식: m_corrected = sqrt(max(m^2 - 2*sigma^2, 0))
    m_nc = np.sqrt(np.maximum(m_n**2 - 2*sigma_global**2, 0))
    m_dc = np.sqrt(np.maximum(m_d**2 - 2*sigma_global**2, 0))
    
    # SNR 및 SSIM 계산 (ground truth 기준)
    snr_b = compute_snr(m_nc, m_o, mask, is_complex=False)  # 보정 전 noisy와 비교
    snr_a = compute_snr(m_dc, m_o, mask, is_complex=False)  # 보정 후 denoised와 비교
    ssim_b = ssim(m_o, m_nc, data_range=np.ptp(m_o), mask=mask)
    ssim_a = ssim(m_o, m_dc, data_range=np.ptp(m_o), mask=mask)
    
    rows.append(dict(Echo=e, SNR_before=snr_b, SNR_after=snr_a, ΔSNR=snr_a-snr_b,
                    SSIM_before=ssim_b, SSIM_after=ssim_a, ΔSSIM=ssim_a-ssim_b))

# DataFrame 생성 및 평균값 추가
df = pd.DataFrame(rows)
average = df.mean(numeric_only=True)
average_row = {col: average[col] for col in df.columns if col != 'Echo'}
average_row['Echo'] = 'Average'
df = pd.concat([df, pd.DataFrame([average_row])], ignore_index=True)

# 출력
print("\nEcho-wise Performance Comparison (Bias-Corrected with Global Sigma):")
print(df.round(3))

# 필요하면 Excel 저장
with pd.ExcelWriter("SNR_SSIM_BiasCorrected_GlobalSigma.xlsx") as writer:
    df.to_excel(writer, sheet_name="BiasCorrected_GlobalSigma", index=False)
print("✔ Excel 저장 완료: SNR_SSIM_BiasCorrected_GlobalSigma.xlsx")
