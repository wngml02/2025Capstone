import warnings

import napari
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from tqdm import tqdm

# 1. .mat 파일 로드
mat_data = sio.loadmat("noisy_meas_gre_dir1.mat")
noisy_real_all = mat_data["noisy_real"]  # shape: (256, 224, 176, 6)
noisy_imag_all = mat_data["noisy_imag"]  # shape: (256, 224, 176, 6)

print("Real shape:", noisy_real_all.shape)
print("Imag shape:", noisy_imag_all.shape)

# 2. MP-PCA 함수 정의 (분산 0 방지 + RuntimeWarning 제거 포함)
def mppca_denoise_4d(data, patch_size=5):
    X, Y, Z, N = data.shape
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (pad, pad), (0, 0)), mode='reflect')

    denoised = np.zeros_like(data)
    counts = np.zeros((X, Y, Z), dtype=np.int32)

    for x in tqdm(range(pad, pad+X), desc="Denoising"):
        for y in range(pad, pad+Y):
            for z in range(pad, pad+Z):
                patch = padded[x-pad:x+pad+1, y-pad:y+pad+1, z-pad:z+pad+1, :]
                vecs = patch.reshape(-1, N)

                # PCA 수행 조건 검사
                if np.var(vecs) == 0 or np.isnan(vecs).any():
                    continue

                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        pca = PCA()
                        pca.fit(vecs)

                    eigvals = pca.singular_values_**2
                    sigma2 = np.median(eigvals)
                    keep = eigvals > sigma2

                    if np.sum(keep) == 0:
                        continue

                    components = pca.components_[keep]
                    vecs_denoised = (vecs @ components.T) @ components
                    patch_denoised = vecs_denoised.reshape(patch.shape)

                    denoised[x-pad, y-pad, z-pad, :] += patch_denoised[pad, pad, pad, :]
                    counts[x-pad, y-pad, z-pad] += 1

                except Exception as e:
                    continue

    # 안전한 평균화
    denoised = np.divide(
        denoised, counts[..., None],
        out=np.zeros_like(denoised),
        where=counts[..., None] != 0
    )

    denoised[np.isnan(denoised)] = 0
    return denoised

# 3. Real / Imaginary 각각 디노이징
denoised_real_all = mppca_denoise_4d(noisy_real_all, patch_size=5)
denoised_imag_all = mppca_denoise_4d(noisy_imag_all, patch_size=5)

sio.savemat("denoised_real.mat", {"denoised_real": denoised_real_all})
sio.savemat("denoised_imag.mat", {"denoised_imag": denoised_imag_all})


print("저장 완료!")

# 4. napari 시각화
viewer = napari.Viewer()

for c in range(6):
    real_d = np.transpose(denoised_real_all[:, :, :, c], (2, 0, 1))
    imag_d = np.transpose(denoised_imag_all[:, :, :, c], (2, 0, 1))

    viewer.add_image(real_d.astype(np.float32), name=f"Denoised Real {c}", colormap="gray", opacity=0.8)
    viewer.add_image(imag_d.astype(np.float32), name=f"Denoised Imag {c}", colormap="gray", opacity=0.8)

napari.run()
