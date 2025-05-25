import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# 데이터 로드
ORIG_MAT = 'meas_gre_dir1.mat'
NOISY_MAT = 'noisy_meas_gre_dir1_30.mat'
DENO_MAT = 'denoised_real_imag_30_sqrt_r3.mat'

# Load original, noisy, and denoised data
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
noisy = sio.loadmat(NOISY_MAT, simplify_cells=True)
den = sio.loadmat(DENO_MAT, simplify_cells=True)

mask = orig['mask_brain'].astype(bool)
orig_cplx = orig['meas_gre'].astype(np.complex64)
noisy_real = noisy['noisy_real'].astype(np.float32)
noisy_imag = noisy['noisy_imag'].astype(np.float32)
den_real = den['den_real'].astype(np.float32)
den_imag = den['den_imag'].astype(np.float32)

# 복소수 생성
noisy_cplx = noisy_real + 1j * noisy_imag
den_cplx = den_real + 1j * den_imag

# Magnitude 계산
mag_noisy = np.abs(noisy_cplx)
mag_den = np.abs(den_cplx)

# SNR 계산 함수 (복소수 & magnitude)
def compute_snr(data, ref, mask, is_complex=True):
    if is_complex:
        mean_signal = np.abs(np.mean(ref[mask]))
        noise_std = np.std(data[mask] - ref[mask])
    else:
        mean_signal = np.mean(ref[mask])
        noise_std = np.std(data[mask] - ref[mask])
    return 20 * np.log10(mean_signal / noise_std)

# 복소수 SNR과 magnitude SNR 비교
snr_cplx_before = compute_snr(noisy_cplx, orig_cplx, mask, is_complex=True)
snr_cplx_after = compute_snr(den_cplx, orig_cplx, mask, is_complex=True)

snr_mag_before = compute_snr(mag_noisy, np.abs(orig_cplx), mask, is_complex=False)
snr_mag_after = compute_snr(mag_den, np.abs(orig_cplx), mask, is_complex=False)

print(f"Complex SNR (before): {snr_cplx_before:.2f} dB")
print(f"Complex SNR (after) : {snr_cplx_after:.2f} dB")
print(f"Magnitude SNR (before): {snr_mag_before:.2f} dB")
print(f"Magnitude SNR (after) : {snr_mag_after:.2f} dB")

# 히스토그램 비교: background(신호 없는 영역)
background_mask = ~mask
plt.figure(figsize=(10,5))
plt.hist(np.abs(noisy_cplx[background_mask]).ravel(), bins=100, alpha=0.5, label='Noisy Complex |z|')
plt.hist(mag_noisy[background_mask].ravel(), bins=100, alpha=0.5, label='Noisy Magnitude')
plt.legend()
plt.title('Histogram: Complex |z| vs Magnitude in Background')
plt.xlabel('Signal Intensity')
plt.ylabel('Frequency')
plt.show()
