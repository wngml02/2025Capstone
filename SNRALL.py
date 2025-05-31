# ─── 기존 코드(그대로) ─────────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

orig_file   = "meas_gre_dir1.mat"
noisy_file  = "noisy_meas_gre_dir1_50.mat"
deno_file   = "denoised_real_imag_50_sqrt_r3.mat"

mask_key, orig_key = "mask_brain", "meas_gre"
noisy_keys, deno_keys = ("noisy_real","noisy_imag"), ("den_real","den_imag")

m_o = sio.loadmat(orig_file, simplify_cells=True)
m_n = sio.loadmat(noisy_file, simplify_cells=True)
m_d = sio.loadmat(deno_file, simplify_cells=True)

mask       = m_o[mask_key].astype(bool)
orig_mag   = np.abs(m_o[orig_key]).astype(np.float32)
noisy_mag  = np.abs(m_n[noisy_keys[0]] + 1j*m_n[noisy_keys[1]]).astype(np.float32)
deno_mag   = np.abs(m_d[deno_keys[0]] + 1j*m_d[deno_keys[1]]).astype(np.float32)

def snr_diff_rician(ref_mag, test_mag, brain_mask):
    snr = []
    for c in range(ref_mag.shape[3]):
        brain_ref  = ref_mag[..., c][brain_mask]
        brain_diff = (test_mag[..., c] - ref_mag[..., c])[brain_mask]

        mu    = brain_ref.mean()
        sigma = brain_diff.std(ddof=1)          # clean-noisy → √2,0.655 보정 X

        s_corr = np.sqrt(max(mu**2 - 2*sigma**2, 0))
        snr.append(s_corr / sigma)
    return np.asarray(snr, float)

# ---- Echo 요약 (콘솔 & Excel Sheet 1) ------------------------------------
snr_noisy = snr_diff_rician(orig_mag, noisy_mag, mask)
snr_deno  = snr_diff_rician(orig_mag, deno_mag,  mask)

df_echo = pd.DataFrame({
    "Echo": np.arange(1, len(snr_noisy)+1),
    "SNR_noisy": snr_noisy,
    "SNR_denoised": snr_deno,
    "Improvement": snr_deno - snr_noisy
}).round(3)

print(df_echo.to_string(index=False))   # ← 기존 df 출력 유지

# ─── 새 함수: 슬라이스×에코 상세 SNR ────────────────────────────────────────
def snr_diff_rician_slice(ref_mag, test_mag, brain_mask):
    """
    반환: DataFrame [Slice, Echo, SNR_noisy/denoised/Improvement]
    """
    X, Y, Z, C = ref_mag.shape
    rows = []
    for z in range(Z):
        m2d = brain_mask[:, :, z]
        if not m2d.any():      # 뇌가 전혀 없는 슬라이스는 건너뜀
            continue
        for c in range(C):
            mu    = ref_mag[:, :, z, c][m2d].mean()
            sigma = (test_mag[:, :, z, c] - ref_mag[:, :, z, c])[m2d].std(ddof=1)
            s_corr = np.sqrt(max(mu**2 - 2*sigma**2, 0))
            rows.append({"Slice": z,
                        "Echo":  c,         # 1-based
                        "SNR":   s_corr / sigma})
    return pd.DataFrame(rows)

# ---- 두 지도 생성 & Improvement 열 추가 ----------------------------------
df_noisy_slice = (
    snr_diff_rician_slice(orig_mag, noisy_mag, mask)
    .rename(columns={"SNR": "SNR_noisy"})
)
df_deno_slice  = (
    snr_diff_rician_slice(orig_mag, deno_mag,  mask)
    .rename(columns={"SNR": "SNR_denoised"})
)

df_slice = (
    df_noisy_slice.merge(df_deno_slice, on=["Slice", "Echo"])
                 .assign(Improvement=lambda d: d["SNR_denoised"] - d["SNR_noisy"])
                 .round(3)
                 # ── 열 순서: Echo, Slice, … ──────────────────────────────
                 .loc[:, ["Echo", "Slice", "SNR_noisy", "SNR_denoised", "Improvement"]]
                 # ── Echo-기준, 그다음 Slice-기준 정렬 ────────────────────
                 .sort_values(["Echo", "Slice"], ignore_index=True)
)

# ─── Excel 저장 (SliceDetail 한 시트만) -----------------------------------
out_xlsx = Path("snr_split_50.xlsx")
with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
    df_slice.to_excel(xw, sheet_name="SliceDetail", index=False)

print(f"\n✔ Excel 저장 완료 → {out_xlsx.resolve()}")