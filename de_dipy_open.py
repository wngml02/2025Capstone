import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# 1. Load real/imag (DIPY-denoised)
real = sio.loadmat("denoised_real_dipy_masked.mat")["denoised_real"]
imag = sio.loadmat("denoised_imag_dipy_masked.mat")["denoised_imag"]

# 2. Load original noisy data for magnitude comparison
noisy_real = sio.loadmat("noisy_real_only.mat")["noisy_real"]
noisy_imag = sio.loadmat("noisy_imag_only.mat")["noisy_imag"]

# 3. Load DIPY-denoised magnitude
mag_denoised = sio.loadmat("denoised_magnitude_dipy_masked.mat")["denoised_magnitude"]

# 4. Calculate noisy magnitude
mag_noisy = np.abs(noisy_real + 1j * noisy_imag)

# 5. Choose mid-slice and channel
Z = real.shape[2] // 2
ch = 0  # first channel

# 6. Slice data
slices = [
    real[:, :, Z, ch],
    imag[:, :, Z, ch],
    mag_noisy[:, :, Z, ch],
    mag_denoised[:, :, Z, ch],
]
titles = ["Real (Denoised)", "Imaginary (Denoised)", "Noisy Magnitude", "Denoised Magnitude"]

# 7. Plot
plt.figure(figsize=(18, 4))
for i, (img, title) in enumerate(zip(slices, titles)):
    plt.subplot(1, 4, i + 1)
    plt.imshow(img, cmap="gray")
    plt.title(title, fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.savefig("dipy_results_comparison.png", dpi=300)
plt.show()
