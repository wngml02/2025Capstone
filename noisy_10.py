import numpy as np
import scipy.io as sio

#--------------------------------------------------------------------------
# 1. 데이터 불러오기
#--------------------------------------------------------------------------
mat_data = sio.loadmat("meas_gre_dir1.mat")
meas_gre = mat_data["meas_gre"]       # shape: (256, 224, 176, 6)
mask = mat_data["mask_brain"]         # shape: (256, 224, 176)

#--------------------------------------------------------------------------
# 2. 전체 평균 신호 강도 계산 (magnitude)
#--------------------------------------------------------------------------
mean_signals = []
for c in range(6):
    vol_complex = meas_gre[:, :, :, c]
    magnitude = np.abs(vol_complex)
    brain_region = magnitude[mask == 1]
    mean_signals.append(np.mean(brain_region))

overall_mean = np.mean(mean_signals)
sigma = overall_mean * 0.5

#--------------------------------------------------------------------------
# 3. 노이즈 추가된 real/imaginary를 넣을 배열
#--------------------------------------------------------------------------
noisy_real_all = np.zeros_like(meas_gre, dtype=np.float32)
noisy_imag_all = np.zeros_like(meas_gre, dtype=np.float32)

#--------------------------------------------------------------------------
# 4. real/imaginary에 추가할 노이즈를 넣을 배열
#--------------------------------------------------------------------------
noise_real_all = np.zeros_like(meas_gre, dtype=np.float32)
noise_imag_all = np.zeros_like(meas_gre, dtype=np.float32)

#--------------------------------------------------------------------------
# 5. 노이즈 추가
#--------------------------------------------------------------------------
for c in range(6):
    vol_complex = meas_gre[:, :, :, c]
    real_vol = np.real(vol_complex)
    imag_vol = np.imag(vol_complex)

    noise_real = np.random.normal(0, sigma, size=real_vol.shape)
    noise_imag = np.random.normal(0, sigma, size=imag_vol.shape)

    noisy_real = real_vol + noise_real * mask.astype(bool)
    noisy_imag = imag_vol + noise_imag * mask.astype(bool)

    # 저장용 배열에 담기
    noisy_real_all[:, :, :, c] = noisy_real
    noisy_imag_all[:, :, :, c] = noisy_imag
    noise_real_all[:, :, :, c] = noise_real
    noise_imag_all[:, :, :, c] = noise_imag

#--------------------------------------------------------------------------
# 6. .mat 파일로 저장
#-------------------------------------------------------------------------
sio.savemat("noisy_meas_gre_dir1_50.mat", {
    "noisy_real": noisy_real_all,
    "noisy_imag": noisy_imag_all,
    "noise_real": noise_real_all,
    "noise_imag": noise_imag_all,
    "te_gre": mat_data["te_gre"],
    "mask_brain": mat_data["mask_brain"],
    "meas_gre": meas_gre
})