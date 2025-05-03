import warnings

import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# 1. Load original, noisy, denoised, mask
mat_data = sio.loadmat("meas_gre_dir1.mat")
orig = np.abs(mat_data["meas_gre"])      
denoised = sio.loadmat("denoised_magnitude_dipy_masked.mat")["denoised_magnitude"]
mask = sio.loadmat("mask_brain.mat")["mask_brain"].astype(bool)

# Noisy magnitude 계산
real_noisy = sio.loadmat("noisy_real_only.mat")["noisy_real"]
imag_noisy = sio.loadmat("noisy_imag_only.mat")["noisy_imag"]
noisy = np.abs(real_noisy + 1j * imag_noisy)

# 2. Slice + channel 기준
Z = orig.shape[2] // 2
ch = 0

ref = orig[:, :, Z, ch]
img_noisy = noisy[:, :, Z, ch]
img_denoised = denoised[:, :, Z, ch]
brain_mask = mask[:, :, Z]

# 3. SNR 계산 함수
def compute_snr(ref, target, mask=None):
    if mask is not None:
        ref = ref[mask]
        target = target[mask]
    noise = target - ref
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signal_power = np.mean(ref**2)
        noise_power = np.mean(noise**2)
        return 10 * np.log10(signal_power / (noise_power + 1e-12))

# 4. SSIM 계산
def compute_ssim(ref, target, mask=None):
    if mask is not None:
        ref = ref * mask
        target = target * mask
    return ssim(ref, target, data_range=ref.max() - ref.min())

# 5. 결과 출력
snr_noisy = compute_snr(ref, img_noisy, brain_mask)
snr_denoised = compute_snr(ref, img_denoised, brain_mask)

ssim_noisy = compute_ssim(ref, img_noisy, brain_mask)
ssim_denoised = compute_ssim(ref, img_denoised, brain_mask)

print(f"✅ Denoised SNR:  {snr_denoised:.2f} dB")
print(f"✅ Denoised SSIM: {ssim_denoised:.4f}")
