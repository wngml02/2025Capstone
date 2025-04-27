import time
import warnings

import napari
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca
from tqdm import tqdm

mat_real = sio.loadmat("noisy_real_only.mat")
real = mat_real["noisy_real"]          # (256,224,176,6)
mat_imag = sio.loadmat("noisy_imag_only.mat")
imag = mat_imag["noisy_imag"]

def denoise_slices(data, label, r=2):
    X, Y, Z, N = data.shape
    out = np.zeros_like(data)
    print(f"\n▶ {label}  slice-by-slice denoising (patch_radius={r})")
    for z in tqdm(range(Z), desc=f"{label} slices", ncols=80):
        slice4d = data[:, :, z, :][:, :, None, :]          # (X,Y,1,N)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_slice = mppca(slice4d, patch_radius=r)
        out[:, :, z, :] = d_slice[:, :, 0, :]
    return np.nan_to_num(out)

den_real = denoise_slices(real, "Real")
den_imag = denoise_slices(imag, "Imag")

sio.savemat("denoised_real_dipy_final.mat", {"denoised_real": den_real})
sio.savemat("denoised_imag_dipy_final.mat", {"denoised_imag": den_imag})

# magnitude & napari
mag = np.abs(den_real + 1j*den_imag)
viewer = napari.Viewer()
for c in tqdm(range(mag.shape[-1]), desc="Show ch", ncols=80):
    viewer.add_image(np.transpose(mag[..., c], (2,0,1)).astype(np.float32),
                    name=f"Mag {c}", colormap="gray", opacity=0.8)
napari.run()
