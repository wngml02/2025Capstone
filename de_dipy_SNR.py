import warnings

import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# 1. Load noisy real/imag and denoised magnitude
mat_noisy_real = sio.loadmat("noisy_real_only.mat")
mat_noisy_imag = sio.loadmat("noisy_imag_only.mat")
noisy_real_all = mat_noisy_real["noisy_real"]
noisy_imag_all = mat_noisy_imag["noisy_imag"]

mat_mag = sio.loadmat("denoised_magnitude_dipy_final.mat")
denoised_magnitude = mat_mag["denoised_magnitude"]

print("✅ Data loaded successfully.")

# 2. Compute noisy magnitude
noisy_magnitude = np.abs(noisy_real_all + 1j * noisy_imag_all)
print("✅ Noisy magnitude computed.")

# 3. SNR 계산
def compute_snr(reference, target):
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - target) ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

snr_value = compute_snr(noisy_magnitude, denoised_magnitude)
print(f"\n✅ Overall SNR: {snr_value:.2f} dB")

# 4. SSIM 계산
# 4D를 3D로 합치기 (X,Y,Z×N)
noisy_magnitude_flat = noisy_magnitude.reshape(noisy_magnitude.shape[0],
                                                noisy_magnitude.shape[1],
                                                noisy_magnitude.shape[2] * noisy_magnitude.shape[3])

denoised_magnitude_flat = denoised_magnitude.reshape(denoised_magnitude.shape[0], denoised_magnitude.shape[1], denoised_magnitude.shape[2] * denoised_magnitude.shape[3])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ssim_value = ssim(noisy_magnitude_flat, denoised_magnitude_flat, data_range=denoised_magnitude_flat.max() - denoised_magnitude_flat.min())

print(f"✅ Overall SSIM: {ssim_value:.4f}")
