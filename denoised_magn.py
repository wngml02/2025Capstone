import re
import time

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import scipy.io as sio
from dipy.denoise.localpca import mppca
from dipy.denoise.noise_estimate import estimate_sigma
from skimage.metrics import structural_similarity as ssim

# 데이터 로드
data = sio.loadmat('noisy_meas_gre_dir1_30.mat')
noisy_real = data['noisy_real']
noisy_imag = data['noisy_imag']
mask = data['mask_brain']
original_complex = data['meas_gre']

# magnitude 계산
magnitude_noisy = np.sqrt(noisy_real**2 + noisy_imag**2)
magnitude_original = np.abs(original_complex)

# 경계 비율 계산
mask_boundary = np.gradient(mask.astype(float))
mask_boundary_mag = np.sqrt(mask_boundary[0]**2 + mask_boundary[1]**2 + mask_boundary[2]**2)
boundary_ratio = np.sum(mask_boundary_mag > 0) / np.sum(mask)
print(f"경계 비율: {boundary_ratio:.4f}")
if boundary_ratio > 0.2:
    print("경계가 많아서 디노이징 오류가 클 수 있음")

# SNR 계산 함수

# 4D 전체 MP-PCA 디노이징
magnitude_denoised = mppca(magnitude_noisy, patch_radius=3, mask=mask)


# 결과 저장
sio.savemat('denoised_meg_30_r3.mat', {
    'magnitude_denoised': magnitude_denoised,
    'mask': mask,
})


