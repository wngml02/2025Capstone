import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# 1. Load noisy real and imag
mat_real = sio.loadmat("noisy_real_only.mat")
real = mat_real["noisy_real"]

mat_imag = sio.loadmat("noisy_imag_only.mat")
imag = mat_imag["noisy_imag"]

print(f"Real shape: {real.shape}, Imag shape: {imag.shape}")

# 2. Compute magnitude
magnitude = np.abs(real + 1j * imag)  # shape: (256, 224, 176, 6)

# 3. Average across channels
mean_magnitude = np.mean(magnitude, axis=-1)  # shape: (256, 224, 176)

# 4. Thresholding to create brain mask
# Percentile-based threshold: adjust 60~70% as needed
threshold_value = np.percentile(mean_magnitude, 60)  
mask_brain = (mean_magnitude > threshold_value).astype(np.uint8)  # 1 = brain, 0 = background

print(f"Threshold value: {threshold_value:.4f}")
print(" Brain mask created.")

# 5. Save mask to .mat file
sio.savemat("mask_brain.mat", {"mask_brain": mask_brain})
print("Brain mask saved as 'mask_brain.mat'.")

# 6. Optional: quick visualization of one slice
slice_index = mean_magnitude.shape[2] // 2  # middle slice
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.title("Mean Magnitude (Mid Slice)")
plt.imshow(mean_magnitude[:,:,slice_index], cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Generated Brain Mask (Mid Slice)")
plt.imshow(mask_brain[:,:,slice_index], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
