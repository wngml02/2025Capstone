import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# 파일 로드
DENO_MAT = "denoised_real_imag_30_sqrt_r3.mat"
ORIG_MAT = "meas_gre_dir1.mat"

den = sio.loadmat(DENO_MAT, simplify_cells=True)
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)

mask = orig['mask_brain'].astype(bool)
den_real = den['den_real'].astype(np.float32)
den_imag = den['den_imag'].astype(np.float32)

# 복소수 및 magnitude
den_cplx = den_real + 1j * den_imag
mag_den = np.abs(den_cplx)

# 1️⃣ 실제 파일의 background 영역 추출
background_mask = ~mask
background_data = mag_den[background_mask]

# 2️⃣ Gaussian noise 시뮬레이션
sigma_sim = np.std(background_data)
sim_real = np.random.normal(0, sigma_sim, size=background_data.shape)
sim_imag = np.random.normal(0, sigma_sim, size=background_data.shape)
sim_mag = np.abs(sim_real + 1j * sim_imag)

# 히스토그램 계산
bins = np.linspace(0, max(background_data.max(), sim_mag.max())*1.1, 100)
hist_file, _ = np.histogram(background_data, bins=bins, density=True)
hist_sim, _ = np.histogram(sim_mag, bins=bins, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# 그래프
plt.figure(figsize=(10,6))
plt.plot(bin_centers, hist_file, label='File Background Magnitude (Rician bias)')
plt.plot(bin_centers, hist_sim, label='Simulated Gaussian Noise (No bias)')
plt.xlabel('Magnitude Intensity')
plt.ylabel('Probability Density')
plt.title('Rician Bias (File) vs Gaussian Noise (Simulated)')
plt.legend()
plt.grid(True)
plt.show()
