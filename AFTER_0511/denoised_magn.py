import numpy as np
import scipy.io as sio

# ---------- ① 디노이징 결과 불러오기 ----------
mat = sio.loadmat('denoised_result_fixed.mat')
real_d = mat['den_real']          # shape: (X,Y,Z,C)
imag_d = mat['den_imag']

# ---------- ② 복소수 재조합 ----------
complex_d = real_d + 1j * imag_d

# ---------- ③ magnitude 계산 ----------
mag_denoised = np.abs(complex_d)  # shape: (X,Y,Z,C)

# ---------- ④ 저장 ----------
sio.savemat('denoised_magnitude.mat', {
    'mag_denoised': mag_denoised,
})

print("✔ 디노이징된 magnitude 계산 완료 → 'denoised_magnitude.mat' 저장됨")
