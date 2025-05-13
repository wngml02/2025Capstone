import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ---------- 데이터 불러오기 ----------
true = np.abs(sio.loadmat('meas_gre_dir1.mat')['meas_gre'])
noisy = sio.loadmat('AFTER_0511/noisy_mag_phase.mat')['magnitude_noisy']
denoised = sio.loadmat('AFTER_0511/denoised_magnitude.mat')['mag_denoised']

slice_z = true.shape[2] // 2  # 중간 Z-slice
fig, axs = plt.subplots(3, 6, figsize=(18, 9))

titles = [f"Echo {i}" for i in range(6)]
row_labels = ['Original', 'Noisy', 'Denoised']

# display range 고정 (공통)
vmin = np.min(true)
vmax = np.max(true)

for echo in range(6):
    axs[0, echo].imshow(true[:, :, slice_z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[1, echo].imshow(noisy[:, :, slice_z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[2, echo].imshow(denoised[:, :, slice_z, echo], cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, echo].set_title(titles[echo])

for row in range(3):
    for col in range(6):
        axs[row, col].axis('off')
    axs[row, 0].set_ylabel(row_labels[row], fontsize=14)

plt.tight_layout()
plt.savefig("comparison_all_echoes_fixed_range.png")
plt.close()
