import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# 1. noisy + denoised 데이터 불러오기
noisy_mat = sio.loadmat("noisy_meas_gre_dir1.mat")
den_real = sio.loadmat("denoised_real_dipy.mat")
den_imag = sio.loadmat("denoised_imag_dipy.mat")

noisy_real_all = noisy_mat["noisy_real"]
noisy_imag_all = noisy_mat["noisy_imag"]

denoised_real_all = den_real["denoised_real"]
denoised_imag_all = den_imag["denoised_imag"]

# 2. magnitude 계산
noisy_mag = np.abs(noisy_real_all + 1j * noisy_imag_all)
denoised_mag = np.abs(denoised_real_all + 1j * denoised_imag_all)

# 3. 평가 대상 slice 선택 (중간 z, 채널 0)
slice_index = noisy_mag.shape[2] // 2
vol_index = 0

noisy_slice = noisy_mag[:, :, slice_index, vol_index]
denoised_slice = denoised_mag[:, :, slice_index, vol_index]

# 4. SNR 계산
def compute_snr(noisy, denoised):
    noise = noisy - denoised
    return 10 * np.log10(np.sum(noisy**2) / np.sum(noise**2))

snr_before = compute_snr(denoised_slice, noisy_slice)
print(f"SNR (before denoising): {snr_before:.2f} dB")

# (옵션) 디노이징 후 SNR
snr_after = compute_snr(noisy_slice, denoised_slice)
print(f"SNR (after denoising): {snr_after:.2f} dB")

# 5. SSIM 계산
ssim_val = ssim(noisy_slice, denoised_slice, data_range=denoised_slice.max() - denoised_slice.min())
print(f"SSIM (vs noisy): {ssim_val:.4f}")

# 6. 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(noisy_slice, cmap='gray')
plt.title('Noisy Magnitude')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_slice, cmap='gray')
plt.title('Denoised Magnitude')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(np.abs(noisy_slice - denoised_slice), cmap='hot')
plt.title('Difference (Error)')
plt.axis('off')

plt.tight_layout()
plt.show()
