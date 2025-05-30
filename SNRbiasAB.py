from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ─── 파일 경로 설정 ─────────────────────────────
ref_file = Path("meas_gre_dir1.mat")
noisy_file = Path("noisy_meas_gre_dir1_30.mat")
den_file = Path("denoised_real_imag_30_sqrt_r3.mat")

# ─── 데이터 로드 ────────────────────────────────
data_ref = sio.loadmat(ref_file)
data_noisy = sio.loadmat(noisy_file)
data_den = sio.loadmat(den_file)

signal_mask = data_ref['mask_brain'].astype(bool)

# ─── 복소수 데이터 로드 및 magnitude 계산 ──────
orig_cplx = data_ref['meas_gre'].astype(np.complex64)
noisy_real = data_noisy['noisy_real'].astype(np.float32)
noisy_imag = data_noisy['noisy_imag'].astype(np.float32)
den_real = data_den['den_real'].astype(np.float32)
den_imag = data_den['den_imag'].astype(np.float32)

noisy_cplx = noisy_real + 1j * noisy_imag
den_cplx = den_real + 1j * den_imag
mag_noisy = np.abs(noisy_cplx)
mag_den = np.abs(den_cplx)

# ─── Noise ROI 생성 (mask 반전) ────────────────
noise_mask = ~signal_mask

# ─── SNR 계산 함수 ─────────────────────────────
def calculate_snr(signal_data, signal_mask, noise_mask, rician_correction=True):
    snr_values = []
    for echo in range(signal_data.shape[3]):
        signal_vals = signal_data[:, :, :, echo][signal_mask]
        noise_vals = signal_data[:, :, :, echo][noise_mask]

        signal_mean = np.mean(signal_vals)
        noise_std = np.std(noise_vals)

        if rician_correction:
            noise_std /= 0.66
        snr = signal_mean / noise_std
        snr_db = 20 * np.log10(snr)
        snr_values.append(snr_db)
    return snr_values

# ─── SNR 계산 ────────────────────────────────
snr_noisy_raw = calculate_snr(mag_noisy, signal_mask, noise_mask, rician_correction=False)
snr_noisy_corr = calculate_snr(mag_noisy, signal_mask, noise_mask, rician_correction=True)
snr_den_raw = calculate_snr(mag_den, signal_mask, noise_mask, rician_correction=False)
snr_den_corr = calculate_snr(mag_den, signal_mask, noise_mask, rician_correction=True)

# ─── 결과 출력 ──────────────────────────────
print("\n📊 SNR (Noisy Data):")
for i, (snr_raw, snr_corr) in enumerate(zip(snr_noisy_raw, snr_noisy_corr)):
    print(f"Echo {i+1}: Raw SNR={snr_raw:.2f} dB | Rician Corrected={snr_corr:.2f} dB")

print("\n📊 SNR (Denoised Data):")
for i, (snr_raw, snr_corr) in enumerate(zip(snr_den_raw, snr_den_corr)):
    print(f"Echo {i+1}: Raw SNR={snr_raw:.2f} dB | Rician Corrected={snr_corr:.2f} dB")
