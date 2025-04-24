import warnings

import napari
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import mppca
from dipy.segment.mask import median_otsu

# 1. .mat 파일 로드
mat_data = sio.loadmat("noisy_meas_gre_dir1.mat")
noisy_real_all = mat_data["noisy_real"]
noisy_imag_all = mat_data["noisy_imag"]

print("Real shape:", noisy_real_all.shape)
print("Imag shape:", noisy_imag_all.shape)

# 2. 마스크 생성 (채널 0 기준으로)
_, mask = median_otsu(noisy_real_all[:, :, :, 0], vol_idx=None, numpass=1)

# 3. DIPY MP-PCA 적용 (real / imag 각각)
def dipy_mppca_denoise(data, patch_radius=2, mask=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = mppca(data, patch_radius=patch_radius, mask=mask)
    result = np.nan_to_num(result)  # NaN 방지
    return result

print("▶ DIPY MP-PCA denoising: real")
denoised_real_all = dipy_mppca_denoise(noisy_real_all, patch_radius=2, mask=mask)

print("▶ DIPY MP-PCA denoising: imag")
denoised_imag_all = dipy_mppca_denoise(noisy_imag_all, patch_radius=2, mask=mask)

# 4. 복소수 결합 및 magnitude 계산
complex_denoised = denoised_real_all + 1j * denoised_imag_all
magnitude_all = np.abs(complex_denoised)

# 5. 저장
sio.savemat("denoised_real_dipy.mat", {"denoised_real": denoised_real_all})
sio.savemat("denoised_imag_dipy.mat", {"denoised_imag": denoised_imag_all})
sio.savemat("denoised_magnitude_dipy.mat", {"denoised_magnitude": magnitude_all})
print("저장 완료!")

# 6. 시각화 (선택)
viewer = napari.Viewer()
for c in range(magnitude_all.shape[-1]):
    slice_ = np.transpose(magnitude_all[:, :, :, c], (2, 0, 1))
    viewer.add_image(slice_.astype(np.float32), name=f"Magnitude {c}", colormap="gray")
napari.run()
