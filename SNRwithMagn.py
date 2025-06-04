

orig_file   = "meas_gre_dir1.mat"              
noisy_file  = "noisy_meas_gre_dir1/noisy_meas_gre_dir1_50.mat"     
deno_file   = "denoised_real_imag_50_sqrt_r2.mat"   

mask_key    = "mask_brain"      
orig_key    = "meas_gre"         
noisy_keys  = ("noisy_real", "noisy_imag")
deno_keys   = ("den_real",  "den_imag")

import numpy as np
import pandas as pd
import scipy.io as sio

m_o = sio.loadmat(orig_file,  simplify_cells=True)
m_n = sio.loadmat(noisy_file, simplify_cells=True)
m_d = sio.loadmat(deno_file,  simplify_cells=True)

mask        = m_o[mask_key].astype(bool)
orig_mag    = np.abs(m_o[orig_key]).astype(np.float32)
noisy_mag   = np.abs(m_n[noisy_keys[0]] + 1j*m_n[noisy_keys[1]]).astype(np.float32)
deno_mag    = np.abs(m_d[deno_keys[0]] + 1j*m_d[deno_keys[1]]).astype(np.float32)

def snr_diff_rician(ref_mag, test_mag, brain_mask):
    snr = []
    for c in range(ref_mag.shape[3]):
        brain_ref   = ref_mag[..., c][brain_mask]
        brain_diff  = (test_mag[..., c] - ref_mag[..., c])[brain_mask]

        mean_signal = brain_ref.mean()          
        std_raw     = brain_diff.std(ddof=1) 

        sigma       = std_raw   # 이 부분 아직 미정
        
        s_corr_sq   = mean_signal**2 - 2 * sigma**2
        s_corr      = np.sqrt(max(s_corr_sq, 0.0))

        snr.append(s_corr / sigma)
    return np.asarray(snr, float)

snr_noisy = snr_diff_rician(orig_mag, noisy_mag, mask)
snr_deno  = snr_diff_rician(orig_mag, deno_mag,  mask)

df = pd.DataFrame({
    "Echo":         np.arange(1, len(snr_noisy)+1),
    "SNR_noisy":    snr_noisy,
    "SNR_denoised": snr_deno,
    "Improvement":  snr_deno - snr_noisy
})
pd.options.display.float_format = "{:,.3f}".format
print(df.to_string(index=False))


import numpy as np

A, sigma, N = 5, 1, 1_000_000       
noise = np.random.normal(0, sigma, N) + 1j*np.random.normal(0, sigma, N)
mag   = np.abs(A + noise)

S_meas = mag.mean()
S_corr = np.sqrt(max(S_meas**2 - 2*sigma**2, 0))
print("편향 전후 오차:", abs(S_meas-A)/A, "→", abs(S_corr-A)/A)