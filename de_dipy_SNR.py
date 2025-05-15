#!/usr/bin/env python3
# pip install numpy scipy scikit-image pandas
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# ─── 로드 ───────────────────────────────────────────────
orig = sio.loadmat("meas_gre_dir1.mat",      simplify_cells=True)
nois = sio.loadmat("noisy_meas_gre_dir1.mat",simplify_cells=True)
den  = sio.loadmat("denoised_real_imag.mat", simplify_cells=True)

mask = orig["mask_brain"].astype(bool)
orig_mag = np.abs(orig["meas_gre"])
nois_mag = np.abs(nois["noisy_real"] + 1j*nois["noisy_imag"])
den_mag  = np.abs(den["den_real"]    + 1j*den["den_imag"])

# ─── SNR & SSIM  (echo-별) ─────────────────────────────
def snr(clean, test):
    sig = clean[mask].mean()
    err = ((clean - test)[mask] ** 2).mean() ** 0.5   # RMSE
    return 20*np.log10(sig/err)

rows = []
for e in range(6):
    clean = orig_mag[..., e]
    noisy = nois_mag[..., e]
    denoi = den_mag[..., e]

    rows.append({
        "echo": e+1,

        "SNR_deno":   snr(clean, denoi),
        "SSIM_deno":  ssim(clean, denoi, data_range=clean.max()-clean.min(), gaussian_weights=True, use_sample_covariance=False)
    })

df = pd.DataFrame(rows)

print(df.round(3))
