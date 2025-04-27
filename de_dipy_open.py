import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# 1. Load data
mat_noisy_real = sio.loadmat("noisy_real_only.mat")
mat_noisy_imag = sio.loadmat("noisy_imag_only.mat")
noisy_real = mat_noisy_real["noisy_real"]
noisy_imag = mat_noisy_imag["noisy_imag"]

mat_denoised_mag = sio.loadmat("denoised_magnitude_dipy_final.mat")
denoised_mag = mat_denoised_mag["denoised_magnitude"]

# 2. Calculate noisy magnitude
noisy_mag = np.abs(noisy_real + 1j * noisy_imag)

# 3. Choose one slice (middle slice)
Z = noisy_mag.shape[2] // 2

# 4. Create figure
n_channels = noisy_mag.shape[-1]
n_cols = 2  # Noisy, Denoised
n_rows = n_channels

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))

# 5. Fill plots
for c in range(n_channels):
    noisy_slice = noisy_mag[:, :, Z, c]
    denoised_slice = denoised_mag[:, :, Z, c]
    
    axs[c, 0].imshow(noisy_slice, cmap='gray')
    axs[c, 0].axis('off')
    axs[c, 0].set_title(f"Ch {c}: Noisy")

    axs[c, 1].imshow(denoised_slice, cmap='gray')
    axs[c, 1].axis('off')
    axs[c, 1].set_title(f"Ch {c}: Denoised")

plt.tight_layout()
plt.show()   # <<<< 저장 없이 바로 화면에 띄우기
