import numpy as np
import scipy.io as sio

# ① 기존 데이터 불러오기
mat = sio.loadmat('AFTER_0511/mag_phase_separated.mat')
magnitude = mat['magnitude']
phase     = mat['phase']

# ② 노이즈 추가 (20%)
noise_level = 0.20
sigma = noise_level * np.mean(magnitude)
np.random.seed(42)  # 재현성 고정
noise = np.random.normal(0, sigma, size=magnitude.shape)
magnitude_noisy = np.clip(magnitude + noise, 0, None)  # 음수 방지

# ③ 복소수 복원
noisy_complex = magnitude_noisy * np.exp(1j * phase)
noisy_real    = np.real(noisy_complex)
noisy_imag    = np.imag(noisy_complex)

# ④ 저장 (여러 배열 한 번에 저장 가능)
sio.savemat('AFTER_0511/noisy_mag_phase.mat', {
    'magnitude_noisy': magnitude_noisy,
    'phase': phase,
    'noisy_real': noisy_real,
    'noisy_imag': noisy_imag,
    'noisy_complex': noisy_complex
})

print("✔ 노이즈 추가된 데이터 저장 완료: 'noisy_mag_phase.mat'")
