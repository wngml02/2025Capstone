import napari
import numpy as np
import scipy.io as sio

# 파일 로드
noisy_data = sio.loadmat('noisy_meas_gre_dir1.mat')
noisy_real = noisy_data['noisy_real']
noisy_imag = noisy_data['noisy_imag']
noisy_mag = np.abs(noisy_real + 1j * noisy_imag)

# napari로 데이터 띄우기
viewer = napari.Viewer()
viewer.add_image(noisy_mag, name='Noisy Magnitude', contrast_limits=[0, np.percentile(noisy_mag, 99)])

# napari에서 수동으로 Noise ROI 생성 후 저장
# 예: Polygon, Paint, Label layer 사용 가능
# 저장된 Noise ROI mask는 numpy 배열로 저장
napari.run()
