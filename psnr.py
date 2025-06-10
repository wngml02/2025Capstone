from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

# ─── 설정 ───────────────────────────────
PATCH_R = 3
NOISE_LEVELS = [10, 20, 30, 40, 50]
ORIG_MAT = "meas_gre_dir1.mat"

# ─── 원본 데이터 로드 ──────────────────
m0 = sio.loadmat(ORIG_MAT, simplify_cells=True)
orig_cplx = m0["meas_gre"].astype(np.complex64)     # (X,Y,Z,C)
orig_mag  = np.abs(orig_cplx)
mask      = m0["mask_brain"].astype(bool)

# ─── PSNR 계산 함수 ────────────────────
def psnr(gt, pred, data_range=1.0):
    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return np.inf
    return 20 * np.log10(data_range / np.sqrt(mse))

# ─── 전체 결과 저장용 리스트 ────────────
all_results = []

# ─── 노이즈 레벨별 PSNR 계산 루프 ───────
for lvl in NOISE_LEVELS:
    split_mat_path = f"denoised_real_imag_{lvl}_sqrt_r{PATCH_R}.mat"
    m_split = sio.loadmat(split_mat_path, simplify_cells=True)
    den_real = m_split["den_real"].astype(np.float32)
    den_imag = m_split["den_imag"].astype(np.float32)
    deno_mag_split = np.abs(den_real + 1j * den_imag)

    for c in range(orig_mag.shape[3]):
        gt    = orig_mag[..., c]
        pred  = deno_mag_split[..., c]
        maxv  = gt.max()
        psnr_ = psnr(gt, pred, data_range=maxv)

        all_results.append({
            "NoiseLevel": lvl,
            "Channel": c,
            "PSNR_Split": psnr_
        })

# ─── 결과 저장 및 출력 ───────────────────
df = pd.DataFrame(all_results)
from ace_tools import display_dataframe_to_user

display_dataframe_to_user("Split 방식 PSNR (10~50)", df)
