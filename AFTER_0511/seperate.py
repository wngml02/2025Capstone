import numpy as np
import scipy.io as sio

# 원본 복소수 데이터 불러오기
mat = sio.loadmat('meas_gre_dir1.mat')
data_cpl = mat['meas_gre']        # complex128, shape=(X, Y, Z, C)

# 분리
magnitude = np.abs(data_cpl)
phase     = np.angle(data_cpl)

# 저장
sio.savemat('mag_phase_separated.mat', {
    'magnitude': magnitude,
    'phase': phase
})
print("✔ magnitude / phase 분리 저장 완료")
