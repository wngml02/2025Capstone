import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca
from tqdm import tqdm  # 진행률 표시용

# ---------- 설정 ----------
PATCH_REAL = 2
PATCH_IMAG = 4
SAVE_PATH  = 'AFTER_0511/denoised_result.mat'   # 결과 저장 경로

# ---------- 데이터 불러오기 ----------
print("▶ 데이터 불러오는 중...")
mat       = sio.loadmat('AFTER_0511/noisy_mag_phase.mat')
real      = mat['noisy_real']        # (X, Y, Z, C)
imag      = mat['noisy_imag']
phase     = mat['phase']

mask_mat  = sio.loadmat('mask_brain.mat')
mask      = mask_mat['mask_brain'].astype(bool)  # (X, Y, Z)

X, Y, Z, C = real.shape
print(f"✔ 데이터 크기: real.shape = {real.shape}, echoes = {C}, Z-slices = {Z}")

# ---------- 결과 배열 초기화 ----------
den_real = np.zeros_like(real)
den_imag = np.zeros_like(imag)

# ---------- 슬라이스별 DIPY 디노이징 ----------
print("▶ DIPY MP-PCA 디노이징 중... (진행 바 표시)")
for z in tqdm(range(Z), desc="Denoising Z-slices"):
    if np.sum(mask[:, :, z]) == 0:
        continue  # 뇌가 없는 z-slice는 건너뜀

    # DIPY는 (X, Y, Z=1, C) 형태 필요
    real_input = real[:, :, z, :][:, :, None, :]
    imag_input = imag[:, :, z, :][:, :, None, :]

    # mask 없이 처리 (뇌 없는 슬라이스만 건너뜀)
    den_real_slice = mppca(real_input, patch_radius=PATCH_REAL)[:, :, 0, :]
    den_imag_slice = mppca(imag_input, patch_radius=PATCH_IMAG)[:, :, 0, :]

    # 결과 저장
    den_real[:, :, z, :] = den_real_slice
    den_imag[:, :, z, :] = den_imag_slice

# ---------- 전체 결과 저장 ----------
sio.savemat(SAVE_PATH, {
    'den_real': den_real,
    'den_imag': den_imag,
    'phase': phase
})
print(f"✔ 디노이징 완료! → '{SAVE_PATH}'에 결과 저장됨")
