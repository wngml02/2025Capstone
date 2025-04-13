import napari
import numpy as np
import scipy.io as sio

# .mat 파일 로드
mat_data = sio.loadmat("noisy_meas_gre_dir1.mat")
noisy_real_all = mat_data["noisy_real"]  # shape: (256, 224, 176, 6)
noisy_imag_all = mat_data["noisy_imag"]  # shape: (256, 224, 176, 6)

viewer = napari.Viewer()

for c in range(6):
    real_vol = noisy_real_all[:, :, :, c]  # (X, Y, Z)
    imag_vol = noisy_imag_all[:, :, :, c]

    # napari-friendly로 변환 (Z, X, Y)
    real_vol_t = np.transpose(real_vol, (2, 0, 1))
    imag_vol_t = np.transpose(imag_vol, (2, 0, 1))
    
    viewer.add_image(real_vol_t.astype(np.float32), name=f"Noisy Real {c}", colormap="gray", opacity=0.8)
    viewer.add_image(imag_vol_t.astype(np.float32), name=f"Noisy Imag {c}", colormap="gray", opacity=0.8)
    
napari.run()