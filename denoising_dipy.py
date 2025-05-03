import time
import warnings

import napari
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca
from skimage.exposure import rescale_intensity
from tqdm import tqdm

# 1. Load noisy real/imag and brain mask
mat_real = sio.loadmat("noisy_real_only.mat")
real = mat_real["noisy_real"]

mat_imag = sio.loadmat("noisy_imag_only.mat")
imag = mat_imag["noisy_imag"]

mat_mask = sio.loadmat("mask_brain.mat")
mask = mat_mask["mask_brain"].astype(bool)  # 1=brain, 0=background

print("✅ Data and mask loaded.")

# 2. Define masked denoising function
def denoise_slices_with_mask(data, mask, label, r=2):
    X, Y, Z, N = data.shape
    out = np.zeros_like(data)
    print(f"\n▶ {label} slice-by-slice DIPY denoising with brain mask (patch_radius={r})")
    
    for z in tqdm(range(Z), desc=f"{label} slices", ncols=80):
        slice4d = data[:, :, z, :][:, :, None, :]  # (X,Y,1,N)
        mask2d = mask[:, :, z]                     # (X,Y)

        if not mask2d.any():
            continue  # Skip empty slices
        
        # Mask 적용: background 제외
        masked_slice = slice4d * mask2d[..., None, None]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_slice = mppca(masked_slice, patch_radius=r)
        
        d_slice = np.nan_to_num(d_slice)
        
        # Mask 부분만 결과에 반영
        out[:, :, z, :] = out[:, :, z, :] + d_slice[:, :, 0, :] * mask2d[..., None]
    
    return out

# 3. Apply DIPY denoising separately
den_real = denoise_slices_with_mask(real, mask, "Real", r=2)
den_imag = denoise_slices_with_mask(imag, mask, "Imag", r=2)

# 4. Magnitude 계산
mag = np.abs(den_real + 1j * den_imag)

# 5. (Optional) Contrast stretching
mag = rescale_intensity(mag, in_range='image', out_range=(0, 1))

# 6. Save 결과
sio.savemat("denoised_real_dipy_masked.mat", {"denoised_real": den_real})
sio.savemat("denoised_imag_dipy_masked.mat", {"denoised_imag": den_imag})
sio.savemat("denoised_magnitude_dipy_masked.mat", {"denoised_magnitude": mag})

print("\n💾 Denoised real, imag, magnitude saved.")

# 7. napari로 시각화
viewer = napari.Viewer()
for c in tqdm(range(mag.shape[-1]), desc="Show channels", ncols=80):
    vol = np.transpose(mag[..., c], (2,0,1))
    viewer.add_image(vol.astype(np.float32),
                    name=f"Masked Mag {c}", colormap="gray", opacity=0.8)
napari.run()
