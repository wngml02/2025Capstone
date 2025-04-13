import napari
import numpy as np
import scipy.io as sio

# 1. 디노이징된 .mat 파일 각각 불러오기
real_mat = sio.loadmat("denoised_real.mat")
imag_mat = sio.loadmat("denoised_imag.mat")

# 2. 실제 데이터 꺼내기
denoised_real_all = real_mat["denoised_real"]  # shape: (256, 224, 176, 6)
denoised_imag_all = imag_mat["denoised_imag"]  # shape: (256, 224, 176, 6)

print("real shape:", denoised_real_all.shape)
print("imag shape:", denoised_imag_all.shape)

# 3. 복소수 결합 후 magnitude 계산
complex_all = denoised_real_all + 1j * denoised_imag_all
magnitude_all = np.abs(complex_all)  # shape: (256, 224, 176, 6)

print("magnitude shape:", magnitude_all.shape)

# 4. napari로 시각화 (Z, X, Y 형태로 전환)
viewer = napari.Viewer()

for c in range(6):
    mag_vol = np.transpose(magnitude_all[:, :, :, c], (2, 0, 1))  # (Z, X, Y)
    viewer.add_image(mag_vol.astype(np.float32), name=f"Magnitude {c}", colormap="gray")

napari.run()

# 5. .mat 파일로 저장
sio.savemat("denoised_magnitude_all.mat", {"denoised_magnitude": magnitude_all})
print("저장 완료")
