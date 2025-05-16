#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

# ── 데이터 로드 ─────────────────────────────────────────────────────────
orig = sio.loadmat("meas_gre_dir1.mat",      simplify_cells=True)
nois = sio.loadmat("noisy_meas_gre_dir1_10.mat", simplify_cells=True)
den  = sio.loadmat("denoised_real_imag_10.mat",  simplify_cells=True)

mask      = orig["mask_brain"].astype(bool)
orig_mag  = np.abs(orig["meas_gre"])
nois_mag  = np.abs(nois["noisy_real"] + 1j*nois["noisy_imag"])
den_mag   = np.abs(den["den_real"]    + 1j*den["den_imag"])

# ── SNR / SSIM 함수 ─────────────────────────────────────────────────────
def snr(clean, test):
    sig  = clean[mask].mean()
    rmse = np.sqrt(((clean - test)[mask]**2).mean())
    return 20*np.log10(sig / rmse)

rows = []
for e in range(6):
    clean = orig_mag[..., e]
    denoi = den_mag[..., e]

    rows.append({
        "echo":       e+1,
        "SNR_deno":   snr(clean, denoi),
        "SSIM_deno":  ssim(clean, denoi,
                        data_range=np.ptp(clean),
                        gaussian_weights=True,
                        use_sample_covariance=False)
    })

df = pd.DataFrame(rows)
print(df.round(3))

# ── 평균 행 추가 / 출력 ─────────────────────────────────────────────────
mean_vals = df[["SNR_deno", "SSIM_deno"]].mean()
print("\nAverage over 6 echoes:")
print(mean_vals.round(3))
