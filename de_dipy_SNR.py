

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from dipy.denoise.noise_estimate import estimate_sigma
from skimage.metrics import structural_similarity as ssim

# ────────────────── 사용자 설정 ──────────────────────────────────────────
ORIG_MAT  = "meas_gre_dir1.mat"              # 원본 복소수 GRE
NOISY_MAT = "noisy_meas_gre_dir1_50.mat"     # 노이즈 추가본
DENO_MAT  = "denoised_real_imag_50_sqrt_r2.mat"  # 디노이즈 결과
EXCEL_OUT = Path("50_snr_ssim_by_echo_slice.xlsx")
# ────────────────────────────────────────────────────────────────────────

# pandas 출력폭·행수 늘리기 (터미널 절단 방지)
pd.set_option("display.max_rows",    2000)
pd.set_option("display.max_columns", None)
pd.set_option("display.width",       200)

# ───── 1. 데이터 로드 ────────────────────────────────────────────────────
print("⋯ MAT 파일 로드 중")
orig = sio.loadmat(ORIG_MAT,  simplify_cells=True)
nois = sio.loadmat(NOISY_MAT, simplify_cells=True)
den  = sio.loadmat(DENO_MAT,  simplify_cells=True)

mask       = orig["mask_brain"].astype(bool)          # (X,Y,Z)
orig_cplx  = orig["meas_gre"].astype(np.complex64)    # (X,Y,Z,echo)
noisy_real = nois["noisy_real"].astype(np.float32)
noisy_imag = nois["noisy_imag"].astype(np.float32)
den_real   = den["den_real"].astype(np.float32)
den_imag   = den["den_imag"].astype(np.float32)

# ───── 2. Magnitude 계산 (√(real²+imag²)) ───────────────────────────────
print("⋯ Magnitude 계산")
mag_orig  = np.sqrt(np.square(orig_cplx.real) + np.square(orig_cplx.imag))
mag_noisy = np.sqrt(np.square(noisy_real)     + np.square(noisy_imag))
mag_den   = np.sqrt(np.square(den_real)       + np.square(den_imag))
n_echoes  = mag_orig.shape[-1]
n_slices  = mag_orig.shape[2]

# ───── 3. SNR 계산 함수 ─────────────────────────────────────────────────
def compute_snr(signal: np.ndarray, reference: np.ndarray,
                m: np.ndarray) -> float:
    """mask m 내부 SNR(dB) = 10·log10(signal_power / noise_power)"""
    noise = signal - reference
    return 10 * np.log10(np.mean(reference[m]**2) / np.mean(noise[m]**2))

# ───── 4-A. Echo 평균 SNR / SSIM ───────────────────────────────────────
print("⋯ Echo 평균 SNR / SSIM 계산")
rows_echo = []
for e in range(n_echoes):
    o, n, d = mag_orig[..., e], mag_noisy[..., e], mag_den[..., e]
    snr_b, snr_a = compute_snr(n, o, mask), compute_snr(d, o, mask)
    ssim_b = ssim(o, n, data_range=np.ptp(o), mask=mask)
    ssim_a = ssim(o, d, data_range=np.ptp(o), mask=mask)
    rows_echo.append(dict(
                        SNR_before=snr_b, SNR_after=snr_a, ΔSNR=snr_a-snr_b,
                        SSIM_before=ssim_b, SSIM_after=ssim_a,
                        ΔSSIM=ssim_a-ssim_b))
df_echo = pd.DataFrame(rows_echo)
print("\n=== Echo 평균 ===")
print(df_echo.round(3))

# ───── 4-B. Echo–Slice SNR / SSIM ──────────────────────────────────────
print("⋯ Echo–Slice 세부 SNR / SSIM 계산")
rows_es = []
for e in range(n_echoes):
    for s in range(n_slices):
        msk = mask[:, :, s]
        if not msk.any():
            continue
        o, n, d = (mag_orig[:, :, s, e],
                   mag_noisy[:, :, s, e],
                   mag_den[:, :, s, e])
        snr_b, snr_a = compute_snr(n, o, msk), compute_snr(d, o, msk)
        ssim_b = ssim(o, n, data_range=np.ptp(o), mask=msk)
        ssim_a = ssim(o, d, data_range=np.ptp(o), mask=msk)
        rows_es.append(dict(Echo=e+1, Slice=s,
                            SNR_before=snr_b, SNR_after=snr_a,
                            ΔSNR=snr_a-snr_b,
                            SSIM_before=ssim_b, SSIM_after=ssim_a,
                            ΔSSIM=ssim_a-ssim_b))
df_es = pd.DataFrame(rows_es)

# ───── 5. Residual σ 추정 & 히스토그램 ─────────────────────────────────
print("⋯ Residual σ 추정 & 히스토그램")
res = mag_noisy - mag_den
σ_before = float(estimate_sigma(mag_noisy[mask].reshape(-1,1,1,1), N=4))
σ_after  = float(estimate_sigma(res[mask].reshape(-1,1,1,1),      N=4))
print(f"σ before: {σ_before:.4f},  after: {σ_after:.4f}")

plt.hist(res[mask].ravel(), bins=100, density=True, alpha=.7)
plt.title("Residual histogram (zero-mean Gaussian/Rician)")
plt.show()

# ───── 6. 엑셀 저장 (시트 이름 ASCII) ───────────────────────────────────
print("⋯ 엑셀 파일 저장")
with pd.ExcelWriter(EXCEL_OUT, engine="xlsxwriter") as xls:
    df_es.round(4).to_excel(xls, sheet_name="Echo_Slice", index=False)
    df_echo.round(4).to_excel(xls, sheet_name="Echo_Mean",  index=False)
    df_echo.mean(numeric_only=True).to_frame("Mean").T.round(4) \
           .to_excel(xls, sheet_name="Summary", index=False)
print(f"\n✔ 엑셀 저장 완료 → {EXCEL_OUT}")
