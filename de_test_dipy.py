import time
import warnings

import napari
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca

# 1. Load noisy real and imag data
mat_real = sio.loadmat("noisy_real_only.mat")
noisy_real_all = mat_real["noisy_real"]

mat_imag = sio.loadmat("noisy_imag_only.mat")
noisy_imag_all = mat_imag["noisy_imag"]

print("Real shape:", noisy_real_all.shape)
print("Imag shape:", noisy_imag_all.shape)

# 2. Denoising function with progress display
def dipy_mppca_denoise_verbose(data, patch_radius=1, label=""):
    print(f"\n▶ Starting DIPY denoising: {label} (patch_radius={patch_radius})")
    start_time = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = mppca(data, patch_radius=patch_radius)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Completed DIPY denoising: {label} ({elapsed_time:.2f} seconds)")
    return np.nan_to_num(result)

# 3. Apply DIPY denoising to real and imag separately
denoised_real_all = dipy_mppca_denoise_verbose(noisy_real_all, patch_radius=1, label="Real")
denoised_imag_all = dipy_mppca_denoise_verbose(noisy_imag_all, patch_radius=1, label="Imaginary")

# 4. Save results
sio.savemat("denoised_real_dipy.mat", {"denoised_real": denoised_real_all})
sio.savemat("denoised_imag_dipy.mat", {"denoised_imag": denoised_imag_all})

print("\n💾 All results saved successfully!")

# 5. Combine real and imag into complex, calculate magnitude
print("\n▶ Generating magnitude images...")
complex_denoised_all = denoised_real_all + 1j * denoised_imag_all
magnitude_all = np.abs(complex_denoised_all)

# 6. Visualize magnitude with napari
print("\n▶ Launching napari viewer...")
viewer = napari.Viewer()

for c in range(magnitude_all.shape[-1]):
    vol = magnitude_all[:, :, :, c]
    vol_t = np.transpose(vol, (2, 0, 1))  # (Z, X, Y) order for napari
    viewer.add_image(vol_t.astype(np.float32),
                    name=f"Magnitude {c}",
                    colormap="gray",
                    opacity=0.8)

napari.run()
