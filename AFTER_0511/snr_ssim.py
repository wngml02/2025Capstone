import numpy as np
import scipy.io as sio

# ---------- 데이터 불러오기 ----------
true = np.abs(sio.loadmat('AFTER_0511/meas_gre_dir1.mat')['meas_gre'])               # (256,224,176,6)
noisy = sio.loadmat('AFTER_0511/noisy_mag_phase.mat')
mag_noisy = noisy.get('magnitude_noisy')

den = sio.loadmat('AFTER_0511/denoised_magnitude.mat')
mag_denoised = den['mag_denoised']                                        # (256,224,176,6)
mask = sio.loadmat('mask_brain.mat')['mask_brain'].astype(bool)  # shape: (256,224,176)

# ---------- 마스크 적용 평균 ----------
true_mean = np.mean(true[mask])
noisy_mean = np.mean(mag_noisy[mask])
den_mean = np.mean(mag_denoised[mask])

# ---------- 스케일 보정 ----------
scale = true_mean / (den_mean + 1e-8)
den_scaled = mag_denoised * scale

# ---------- SNR 계산 함수 (마스크 적용) ----------
def compute_snr_masked(ref, test, mask):
    noise = test - ref
    return 20 * np.log10(np.mean(ref[mask]) / (np.std(noise[mask]) + 1e-8))

# ---------- 계산 ----------
snr_noisy = compute_snr_masked(true, mag_noisy, mask)
snr_denoised = compute_snr_masked(true, den_scaled, mask)

print("📊 SNR 결과 (뇌 영역 마스크 기반)")
print(f"🔹 Noisy     : {snr_noisy:.2f} dB")
print(f"🔹 Denoised  : {snr_denoised:.2f} dB")



print("mean true   :", np.mean(true_mean))
print("mean noisy  :", np.mean(noisy_mean))
print("mean denois:", np.mean(den_mean))

print("std  noisy  :", np.std(noisy_mean))
print("std  denois:", np.std(den_mean))

print("NaN in denoise:", np.isnan(den_mean).sum())