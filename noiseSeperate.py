import numpy as np
import scipy.io as sio

# 1. noisy .mat 파일 불러오기
mat = sio.loadmat("noisy_meas_gre_dir1.mat")

noisy_real_all = mat["noisy_real"]  # (256, 224, 176, 6)
noisy_imag_all = mat["noisy_imag"]  # (256, 224, 176, 6)

print("Real shape:", noisy_real_all.shape)
print("Imag shape:", noisy_imag_all.shape)

# 2. 다시 저장 (real과 imag 각각)
sio.savemat("noisy_real_only.mat", {"noisy_real": noisy_real_all})
sio.savemat("noisy_imag_only.mat", {"noisy_imag": noisy_imag_all})

print("noisy_real_only.mat 와 noisy_imag_only.mat 저장 완료!")
