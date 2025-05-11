import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca
from tqdm import tqdm

# ---------- 설정 ----------
PATCH_REAL = 2
PATCH_IMAG = 4
SAVE_PATH  = 'AFTER_0511/denoised_result_fixed.mat'

# ---------- 데이터 불러오기 ----------
mat = sio.loadmat('AFTER_0511/noisy_mag_phase.mat')
real = mat['noisy_real']
imag = mat['noisy_imag']
phase = mat['phase']
mask = sio.loadmat('mask_brain.mat')['mask_brain'].astype(bool)

X, Y, Z, C = real.shape
den_real = np.zeros_like(real)
den_imag = np.zeros_like(imag)

# ---------- 디노이징 ----------
print("▶ DIPY MP-PCA 디노이징 중...")
for z in tqdm(range(Z), desc="Denoising Z-slices"):
    if np.sum(mask[:, :, z]) == 0:
        continue

    real_input = real[:, :, z, :][:, :, None, :]
    imag_input = imag[:, :, z, :][:, :, None, :]
    mask_3d = mask[:, :, z][:, :, None]  # DIPY requires 3D mask

    den_real_slice = mppca(real_input, patch_radius=PATCH_REAL, mask=mask_3d)[:, :, 0, :]
    den_imag_slice = mppca(imag_input, patch_radius=PATCH_IMAG, mask=mask_3d)[:, :, 0, :]

    den_real[:, :, z, :] = den_real_slice
    den_imag[:, :, z, :] = den_imag_slice

# ---------- 저장 ----------
sio.savemat(SAVE_PATH, {
    'den_real': den_real,
    'den_imag': den_imag,
    'phase': phase
})
print(f"✔ 음수 보존 + 마스크 적용 디노이징 완료 → '{SAVE_PATH}' 저장됨")
