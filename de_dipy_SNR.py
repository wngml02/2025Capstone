#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dipy.denoise.noise_estimate import estimate_sigma
from skimage.metrics import structural_similarity as ssim

# ── 데이터 로드 ─────────────────────────────────────────────────────────
orig = sio.loadmat("meas_gre_dir1.mat",      simplify_cells=True)
nois = sio.loadmat("noisy_meas_gre_dir1_10.mat", simplify_cells=True)
den  = sio.loadmat("denoised_real_imag_10_dn1.mat",  simplify_cells=True)

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


res = nois_mag - den_mag
nois_roi = nois_mag[mask].copy().reshape(-1, 1, 1, 1)  # 4-D 형식 강제
res_roi  = res[mask].copy().reshape(-1, 1, 1, 1)

sigma_before = float(estimate_sigma(nois_roi, N=4))
sigma_after  = float(estimate_sigma(res_roi,  N=4))
print(f"σ before: {sigma_before:.4f},  after: {sigma_after:.4f}")

# 히스토그램
plt.hist(res[mask].ravel(), bins=100, density=True, alpha=.7)
plt.title("Residual histogram (should be zero-mean Gaussian/Rician)")
plt.show()

# ── 평균 행 추가 / 출력 ─────────────────────────────────────────────────
mean_vals = df[["SNR_deno", "SSIM_deno"]].mean()
print("\nAverage over 6 echoes:")
print(mean_vals.round(3))
