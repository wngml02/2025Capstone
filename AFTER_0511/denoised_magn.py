import os

import matplotlib.pyplot as plt
import napari
import numpy as np
import scipy.io as sio

# ---------- 파일 로드 ----------
true = np.abs(sio.loadmat('meas_gre_dir1.mat')['meas_gre'])  # 복소수 → magnitude
noisy = sio.loadmat('AFTER_0511/noisy_mag_phase.mat')['magnitude_noisy']
den = sio.loadmat('AFTER_0511/denoised_magnitude.mat')['mag_denoised']

SLICE_Z = true.shape[2] // 2  # 중간 Z-slice 기준
vmin = np.min(true)
vmax = np.max(true)

# ---------- Napari 시각화 ----------
viewer = napari.Viewer()
for echo in range(6):
    viewer.add_image(true[:, :, SLICE_Z, echo], name=f'True Echo {echo}', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(noisy[:, :, SLICE_Z, echo], name=f'Noisy Echo {echo}', colormap='gray', contrast_limits=[vmin, vmax])
    viewer.add_image(den[:, :, SLICE_Z, echo], name=f'Denoised Echo {echo}', colormap='gray', contrast_limits=[vmin, vmax])
napari.run()

# ---------- 18분할 이미지 저장 ----------
os.makedirs("comparison_18views", exist_ok=True)
fig, axs = plt.subplots(3, 6, figsize=(18, 9))
row_labels = ['Original', 'Noisy', 'Denoised']
titles = [f"Echo {i}" for i in range(6)]

for echo in range(6):
    axs[0, echo].imshow(true[:, :, SLICE_Z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[1, echo].imshow(noisy[:, :, SLICE_Z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[2, echo].imshow(den[:, :, SLICE_Z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, echo].set_title(titles[echo])

for row in range(3):
    for col in range(6):
        axs[row, col].axis('off')
    axs[row, 0].set_ylabel(row_labels[row], fontsize=14)

plt.tight_layout()
plt.savefig("comparison_18views/comparison_all_echoes_fixed_range.png")
plt.close()
print("✔ napari 시각화 완료 + 18분할 이미지 저장 완료")
