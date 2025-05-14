import matplotlib.pyplot as plt
import numpy as np
import scipy.io

mat = scipy.io.loadmat('meas_gre_dir1.mat')  # 파일 경로에 따라 수정

# 2. 포함된 변수 목록 확인
print("포함된 변수 목록:")
for key in mat.keys():
    if not key.startswith('__'):
        print(f"- {key}: shape {mat[key].shape}, dtype {mat[key].dtype}")

