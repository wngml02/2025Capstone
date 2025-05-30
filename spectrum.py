import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from numpy.fft import fft

# ─── 파일 로드 ─────────────────────────────
ORIG_MAT = "meas_gre_dir1.mat"
NOISY_MAT = "noisy_meas_gre_dir1_30.mat"
DENO_MAT = "denoised_real_imag_30_sqrt_r3.mat"

print("⋯ 데이터 로드")
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
noisy = sio.loadmat(NOISY_MAT, simplify_cells=True)
den = sio.loadmat(DENO_MAT, simplify_cells=True)

# 데이터 준비
orig_cplx = orig['meas_gre'].astype(np.complex64)
noisy_real = noisy['noisy_real'].astype(np.float32)
noisy_imag = noisy['noisy_imag'].astype(np.float32)
den_real = den['den_real'].astype(np.float32)
den_imag = den['den_imag'].astype(np.float32)

# 복소수 변환
noisy_cplx = noisy_real + 1j * noisy_imag
den_cplx = den_real + 1j * den_imag

# ─── 스펙트럼 시각화 (노이즈:막대 + 디노이즈:꺾은선) ─────────────────
def plot_combined_spectrum(noisy_vol, den_vol, title, slice_idx=50, axis=0):
    # Slice 추출
    noisy_slice = noisy_vol[:, :, slice_idx]
    den_slice = den_vol[:, :, slice_idx]
    
    if axis == 0:
        noisy_line = noisy_slice[noisy_slice.shape[0]//2, :]
        den_line = den_slice[den_slice.shape[0]//2, :]
    else:
        noisy_line = noisy_slice[:, noisy_slice.shape[1]//2]
        den_line = den_slice[:, den_slice.shape[1]//2]
    
    # FFT
    fft_noisy = np.abs(fft(noisy_line))
    fft_den = np.abs(fft(den_line))
    freq = np.fft.fftfreq(len(noisy_line))
    
    # 시각화
    plt.figure(figsize=(10,6))
    plt.bar(freq, fft_noisy, width=0.002, alpha=0.6, label='Noisy Spectrum', color='orange')
    plt.plot(freq, fft_den, color='blue',label='Denoised Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title(f"{title} Spectrum (Slice {slice_idx}, {'Row' if axis==0 else 'Col'})")
    plt.grid(True)
    plt.legend()
    plt.show()

# ─── 예제: echo=0, slice=50 ─────────────────────────────
echo_idx = 0
slice_idx = 88

print("⋯ 스펙트럼 (노이즈:막대 + 디노이즈:꺾은선) 시각화 중")
plot_combined_spectrum(noisy_cplx[..., echo_idx], den_cplx[..., echo_idx], "Noisy vs Denoised", slice_idx, axis=0)
