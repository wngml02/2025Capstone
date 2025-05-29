from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ─── 파일 경로 설정 ─────────────────────────────
ref_file = Path("meas_gre_dir1.mat")             # Ground Truth
noisy_file = Path("noisy_meas_gre_dir1_30.mat")     # 노이즈 포함
den_file = Path("denoised_real_imag_30_sqrt_r3.mat")    # Denoised

# ─── 데이터 로드 ────────────────────────────────
data_ref = sio.loadmat(ref_file)
data_noisy = sio.loadmat(noisy_file)
data_den = sio.loadmat(den_file)

ref = data_ref['meas_gre']
noisy = data_noisy['noisy_meas_gre']
den = data_den['den_meas_gre']
mask = data_ref['mask_brain']

# ─── 노이즈 표준편차 추정 (마스크 내에서) ────────
sigma_est = np.std((noisy - ref)[mask == 1])
print(f"Estimated noise sigma: {sigma_est:.4f}")

# ─── MSE & RMSE 계산 (Rician 보정 비교) ───────────
def compute_mse_rmse_with_rician(ref, comp, mask, sigma):
    mse_no_corr, rmse_no_corr = [], []
    mse_corr, rmse_corr = [], []
    for echo in range(ref.shape[3]):
        mse_echo_no, rmse_echo_no = [], []
        mse_echo_corr, rmse_echo_corr = [], []
        for slice_idx in range(ref.shape[2]):
            ref_slice = ref[:, :, slice_idx, echo]
            comp_slice = comp[:, :, slice_idx, echo]
            mask_slice = mask[:, :, slice_idx]

            # ── 보정 없는 diff ──
            diff_no = (ref_slice - comp_slice)[mask_slice == 1]
            mse_no = np.mean(np.square(diff_no))
            rmse_no = np.sqrt(mse_no)

            # ── Rician 보정 적용 ──
            comp_corr = np.sqrt(np.maximum(np.square(comp_slice) - 2 * sigma**2, 0))
            ref_corr = np.sqrt(np.maximum(np.square(ref_slice) - 2 * sigma**2, 0))
            diff_corr = (ref_corr - comp_corr)[mask_slice == 1]
            mse_c = np.mean(np.square(diff_corr))
            rmse_c = np.sqrt(mse_c)

            mse_echo_no.append(mse_no)
            rmse_echo_no.append(rmse_no)
            mse_echo_corr.append(mse_c)
            rmse_echo_corr.append(rmse_c)
        mse_no_corr.append(mse_echo_no)
        rmse_no_corr.append(rmse_echo_no)
        mse_corr.append(mse_echo_corr)
        rmse_corr.append(rmse_echo_corr)
    return mse_no_corr, rmse_no_corr, mse_corr, rmse_corr

# ─── 보정 전/후 계산 ─────────────────────────────
mse_noisy_no, rmse_noisy_no, mse_noisy_corr, rmse_noisy_corr = compute_mse_rmse_with_rician(ref, noisy, mask, sigma_est)
mse_den_no, rmse_den_no, mse_den_corr, rmse_den_corr = compute_mse_rmse_with_rician(ref, den, mask, sigma_est)

# ─── 출력 예시 ───────────────────────────────────
for echo in range(6):
    print(f"\nEcho {echo+1}:")
    for slice_idx in range(ref.shape[2]):
        print(f"  Slice {slice_idx+1}: "
              f"[Noisy] MSE={mse_noisy_no[echo][slice_idx]:.6f}, RMSE={rmse_noisy_no[echo][slice_idx]:.6f} | "
              f"MSE_corr={mse_noisy_corr[echo][slice_idx]:.6f}, RMSE_corr={rmse_noisy_corr[echo][slice_idx]:.6f}")
        print(f"          [Denoised] MSE={mse_den_no[echo][slice_idx]:.6f}, RMSE={rmse_den_no[echo][slice_idx]:.6f} | "
              f"MSE_corr={mse_den_corr[echo][slice_idx]:.6f}, RMSE_corr={rmse_den_corr[echo][slice_idx]:.6f}")
