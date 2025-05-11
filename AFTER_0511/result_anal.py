import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ---------- 설정 ----------
SLICE_Z = 88
SAVE_DIR = "AFTER_0511/echo_comparison_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 데이터 불러오기 ----------
meas = sio.loadmat("AFTER_0511/meas_gre_dir1.mat")['meas_gre']
mag_true = np.abs(meas)

noisy = sio.loadmat("AFTER_0511/noisy_mag_phase.mat")
mag_noisy = noisy['magnitude_noisy']

den = sio.loadmat("AFTER_0511/denoised_magnitude.mat")
mag_denoised = den['mag_denoised']

# ---------- echo 0~5 추출 ----------
def get_slices(mag):
    return [mag[:, :, SLICE_Z, i] for i in range(6)]

slices_true = get_slices(mag_true)
slices_noisy = get_slices(mag_noisy)
slices_denoised = get_slices(mag_denoised)

# ---------- 이미지 그리기 ----------
fig, axs = plt.subplots(3, 6, figsize=(18, 9))

titles = [f"Echo {i}" for i in range(6)]
row_labels = ['Original', 'Noisy', 'Denoised']

for col in range(6):
    axs[0, col].imshow(slices_true[col], cmap='gray')
    axs[1, col].imshow(slices_noisy[col], cmap='gray')
    axs[2, col].imshow(slices_denoised[col], cmap='gray')
    axs[0, col].set_title(titles[col], fontsize=12)

for row in range(3):
    for col in range(6):
        axs[row, col].axis('off')
    axs[row, 0].set_ylabel(row_labels[row], fontsize=14)

plt.tight_layout()
save_path = os.path.join(SAVE_DIR, "echo_0_to_5_comparison.png")
plt.savefig(save_path)
plt.close()

print(f"✔ 모든 echo를 한 장에 저장 완료: {save_path}")
