import napari
import numpy as np
import scipy.io as sio

# ---------- ① denoised magnitude 불러오기 ----------
mag_mat = sio.loadmat('AFTER_0511/denoised_magnitude.mat')
mag_denoised = mag_mat['mag_denoised']    # shape: (256,224,176,6)

# ---------- ② 원본 phase 불러오기 (from meas_gre_dir1.mat) ----------
meas_mat = sio.loadmat('meas_gre_dir1.mat')
meas_gre = meas_mat['meas_gre']           # complex128
phase_original = np.angle(meas_gre)       # shape: same

# ---------- ③ 복소수 재조합 ----------
reconstructed = mag_denoised * np.exp(1j * phase_original)

# ---------- ④ 시각화 (echo 0) ----------
echo_index = 0
mag_view = mag_denoised[..., echo_index]         # shape: (256,224,176)
mag_view_napari = np.transpose(mag_view, (2,1,0)) # shape: (Z,Y,X)

viewer = napari.Viewer()
viewer.add_image(mag_denoised, name="denoised", colormap='gray', contrast_limits=(0, 100))
napari.run()

# ---------- ⑤ 저장 ----------
sio.savemat('AFTER_0511/reconstructed_complex_with_original_phase.mat', {
    'reconstructed': reconstructed
})

print("✔ 복소수 재조합 완료 → 'reconstructed_complex_with_original_phase.mat' 저장됨")
