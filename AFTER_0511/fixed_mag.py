import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ---------- 파일 불러오기 ----------
den = sio.loadmat('AFTER_0511/denoised_result_fixed.mat')
real_d = den['den_real']  # shape: (X, Y, Z, C)
imag_d = den['den_imag']
phase  = den['phase']

# ---------- 복소수 재조합 및 magnitude 계산 ----------
den_complex = real_d + 1j * imag_d
mag_denoised = np.abs(den_complex)

# ---------- 저장 ----------
sio.savemat('denoised_magnitude.mat', {
    'mag_denoised': mag_denoised,
    'phase': phase
})
print("✔ denoised_magnitude.mat 저장 완료")

# ---------- echo 평균 및 시각화 ----------
mag_mean = np.mean(mag_denoised, axis=3)
SLICE_Z = mag_mean.shape[2] // 2
vmin = np.min(mag_mean)
vmax = np.max(mag_mean)

plt.figure(figsize=(6, 6))
plt.imshow(mag_mean[:, :, SLICE_Z], cmap='gray', vmin=vmin, vmax=vmax)
plt.title("Denoised Magnitude (Average over Echoes)")
plt.axis('off')
plt.tight_layout()
os.makedirs("denoised_magnitude_display", exist_ok=True)
plt.savefig("denoised_magnitude_display/denoised_magnitude_avg_echo_fixed_range.png")
plt.close()
print("✔ 시각화 이미지 저장 완료 (고정 display range)")
