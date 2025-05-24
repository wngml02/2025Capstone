from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dipy.denoise.noise_estimate import estimate_sigma
from skimage.metrics import structural_similarity as ssim

# ---------------- 사용자 설정 -------------------------------------------
ORIG_MAT  = "meas_gre_dir1.mat"         # 원본 복소수 GRE
NOISY_MAT = "noisy_meas_gre_dir1_10.mat"  # 노이즈 추가본
DENO_MAT  = "denoised_real_imag_10_sqrt_r2.mat"  # 디노이즈 결과
EXCEL_OUT = "10_snr_ssim_by_echo_slice.xlsx"      # 결과 엑셀
Z_HIST    = False   # True → slice별 hist, False → 전체 residual hist
# ------------------------------------------------------------------------

print("⋯ MAT 파일 로드 중")
orig = sio.loadmat(ORIG_MAT,  simplify_cells=True)
nois = sio.loadmat(NOISY_MAT, simplify_cells=True)
den  = sio.loadmat(DENO_MAT,  simplify_cells=True)

mask       = orig["mask_brain"].astype(bool)
orig_cplx  = orig["meas_gre"].astype(np.complex64)   # (X,Y,Z,echo)
noisy_real = nois["noisy_real"].astype(np.float32)
noisy_imag = nois["noisy_imag"].astype(np.float32)
den_real   = den["den_real"].astype(np.float32)
den_imag   = den["den_imag"].astype(np.float32)

# ---------- Magnitude = sqrt(real^2 + imag^2) ---------------------------
print("⋯ Magnitude 계산")
mag_orig  = np.sqrt(np.square(orig_cplx.real) + np.square(orig_cplx.imag))
mag_noisy = np.sqrt(np.square(noisy_real)     + np.square(noisy_imag))
mag_den   = np.sqrt(np.square(den_real)       + np.square(den_imag))

n_echoes  = mag_orig.shape[-1]

# ---------- SNR 계산 함수 -----------------------------------------------
def compute_snr(signal: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """mask 내부 SNR(dB) = 10·log10(signal_p / noise_p)"""
    noise = signal - reference
    signal_power = np.mean(reference[mask] ** 2)
    noise_power  = np.mean(noise[mask]    ** 2)
    return 10 * np.log10(signal_power / noise_power)

# ---------- Echo 평균 테이블 -------------------------------------------
print("⋯ Echo 평균 SNR / SSIM 계산")
rows_echo = []
for echo in range(n_echoes):
    o = mag_orig[..., echo];  n = mag_noisy[..., echo];  d = mag_den[..., echo]
    snr_b = compute_snr(n, o, mask);  snr_a = compute_snr(d, o, mask)
    ssim_b = ssim(o, n, data_range=np.ptp(o), mask=mask)
    ssim_a = ssim(o, d, data_range=np.ptp(o), mask=mask)
    rows_echo.append({
        "Echo": echo+1,
        "SNR_before": snr_b,  "SNR_after": snr_a,  "ΔSNR": snr_a - snr_b,
        "SSIM_before": ssim_b, "SSIM_after": ssim_a, "ΔSSIM": ssim_a - ssim_b
    })

df_echo = pd.DataFrame(rows_echo)
print(df_echo.round(3))

# ---------- Echo‑Slice 테이블 ------------------------------------------
print("⋯ Echo–Slice 세부 SNR / SSIM 계산")
rows_es, n_slices = [], mag_orig.shape[2]
for echo in range(n_echoes):
    for s in range(n_slices):
        msk = mask[:, :, s]
        if not msk.any():
            continue
        o = mag_orig[:, :, s, echo];  n = mag_noisy[:, :, s, echo];  d = mag_den[:, :, s, echo]
        snr_b = compute_snr(n, o, msk);  snr_a = compute_snr(d, o, msk)
        ssim_b = ssim(o, n, data_range=np.ptp(o), mask=msk)
        ssim_a = ssim(o, d, data_range=np.ptp(o), mask=msk)
        rows_es.append({
            "Echo": echo+1, "Slice": s,
            "SNR_before": snr_b, "SNR_after": snr_a, "ΔSNR": snr_a - snr_b,
            "SSIM_before": ssim_b, "SSIM_after": ssim_a, "ΔSSIM": ssim_a - ssim_b
        })

df_es = pd.DataFrame(rows_es)

# ---------- Residual σ 추정 & 히스토그램 -------------------------------
print("⋯ Residual σ 추정 & 히스토그램")
res = mag_noisy - mag_den
sigma_before = float(estimate_sigma(mag_noisy[mask].reshape(-1,1,1,1), N=4))
sigma_after  = float(estimate_sigma(res[mask].reshape(-1,1,1,1),      N=4))
print(f"σ before: {sigma_before:.4f},  after: {sigma_after:.4f}")

plt.hist(res[mask].ravel(), bins=100, density=True, alpha=.7)
plt.title("Residual histogram (zero‑mean Gaussian/Rician)")
plt.show()

# ---------- 엑셀 저장 ----------------------------------------------------
print("⋯ 엑셀 파일 저장")
with pd.ExcelWriter(EXCEL_OUT, engine="xlsxwriter") as writer:
    df_es.round(4).to_excel(writer, sheet_name="Echo‑Slice", index=False)
    df_echo.round(4).to_excel(writer, sheet_name="Echo‑Mean",  index=False)
    df_echo.mean(numeric_only=True).to_frame("Mean").T.round(4) \
           .to_excel(writer, sheet_name="Summary", index=False)
print(f"✔ 엑셀 저장 완료 → {EXCEL_OUT}")
