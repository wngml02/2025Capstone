import napari
import numpy as np
import scipy.io as sio

# denoised_real_all.mat 파일 불러오기
real_mat = sio.loadmat("denoised_real.mat")

# 실제 데이터 꺼내기
denoised_real_all = real_mat["denoised_real"]

print("denoised_real_all shape:", denoised_real_all.shape)


viewer = napari.Viewer()

for c in range(6):
    real_vol = np.transpose(denoised_real_all[:, :, :, c], (2, 0, 1))  # (Z, X, Y)
    viewer.add_image(real_vol.astype(np.float32), name=f"Denoised Real {c}", colormap="gray")


napari.run()