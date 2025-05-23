import napari
import numpy as np
import scipy.io as sio

# 1. 데이터 불러오기
mat_data = sio.loadmat("meas_gre_dir1.mat")
meas_gre = mat_data["meas_gre"]       # shape: (256, 224, 176, 6)
mask = mat_data["mask_brain"]         # shape: (256, 224, 176)

# 2. 전체 평균 신호 강도 계산
mean_signals = []
for c in range(6):
    real_vol = np.real(meas_gre[:, :, :, c])
    brain_region = real_vol[mask == 1]
    mean_signals.append(np.mean(brain_region))

overall_mean = np.mean(mean_signals)
sigma = overall_mean * 0.1
print(f"전체 채널 평균 신호 강도: {overall_mean:.4f}")
print(f"노이즈 표준편차 (σ): {sigma:.4f}")

# transpose → napari-friendly (Z, Y, X)
mask = np.transpose(mask, (2, 0, 1))  # shape: (176, 256, 224)

# 3. napari 시각화
viewer = napari.Viewer()

# 4. 노이즈 추가된 real/imaginary 전체를 저장용 배열에 담기
noisy_real_all = np.zeros_like(meas_gre, dtype=np.float32)
noisy_imag_all = np.zeros_like(meas_gre, dtype=np.float32)

for c in range(6):
    vol_complex = meas_gre[:, :, :, c]
    real_vol = np.real(vol_complex)
    imag_vol = np.imag(vol_complex)

    real_vol_t = np.transpose(real_vol, (2, 0, 1))
    imag_vol_t = np.transpose(imag_vol, (2, 0, 1))

    noise_real = np.random.normal(0, sigma, size=real_vol_t.shape)
    noise_imag = np.random.normal(0, sigma, size=imag_vol_t.shape)

    noisy_real = real_vol_t + noise_real * mask
    noisy_imag = imag_vol_t + noise_imag * mask

    # 다시 원래 shape로 되돌리기 (Z, X, Y) → (X, Y, Z)
    noisy_real_back = np.transpose(noisy_real, (1, 2, 0))
    noisy_imag_back = np.transpose(noisy_imag, (1, 2, 0))

    # 저장용 배열에 담기
    noisy_real_all[:, :, :, c] = noisy_real_back
    noisy_imag_all[:, :, :, c] = noisy_imag_back

    # napari 시각화용
    viewer.add_image(real_vol_t.astype(np.float32), name=f"Real {c}", colormap='gray', opacity=0.8)
    viewer.add_image(imag_vol_t.astype(np.float32), name=f"Imag {c}", colormap='gray', opacity=0.8)
    viewer.add_image(noisy_real.astype(np.float32), name=f"Noisy Real {c}", colormap='gray', opacity=0.8)
    viewer.add_image(noisy_imag.astype(np.float32), name=f"Noisy Imag {c}", colormap='gray', opacity=0.8)
    
    diff_real = np.abs(noisy_real - real_vol_t)
    viewer.add_image(diff_real, name="Difference Real", colormap="gray")
    diff_real = np.abs(noisy_real - real_vol_t)
    viewer.add_image(diff_real, name="Difference Real", colormap="gray")

viewer.add_labels(mask, name="Brain Mask")

# 5. .mat 파일로 저장
sio.savemat("noisy_meas_gre_dir1.mat", {
    "noisy_real": noisy_real_all,
    "noisy_imag": noisy_imag_all
})

napari.run()