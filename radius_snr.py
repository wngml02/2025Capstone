
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

R_LIST      = [1, 2, 3]               # 비교할 patch radius
ORIG_MAT    = "meas_gre_dir1.mat"
DEN_TMPL    = "denoised_real_imag_r{}.mat"   # r 값은 {}에 포맷
MASK_KEY, CPLX_KEY = "mask_brain", "meas_gre"

# ─── 원본 · 마스크 로드 ────────────────────────────────────────────────────
orig = sio.loadmat(ORIG_MAT, simplify_cells=True)
orig_mag = np.abs(orig[CPLX_KEY])                # (X,Y,Z,6)
mask = orig[MASK_KEY].astype(bool)

def snr(clean, test):
    sig = clean[mask].mean()
    rmse = np.sqrt(np.mean((clean - test)[mask]**2))
    return 20*np.log10(sig/rmse)

rows = []
for r in R_LIST:
    mat = sio.loadmat(DEN_TMPL.format(r), simplify_cells=True)
    den_mag = np.abs(mat["den_real"] + 1j*mat["den_imag"])

    for e in range(6):
        clean = orig_mag[..., e]
        den   = den_mag[..., e]
        rows.append({
            "radius": r,
            "echo":   e+1,
            "SNR":  snr(clean, den),
            "SSIM": ssim(clean, den,
                         data_range=np.ptp(clean),
                         gaussian_weights=True,
                         use_sample_covariance=False)
        })

df = pd.DataFrame(rows)

# ─── 콘솔 출력 ────────────────────────────────────────────────────────────
print("\n============= SNR · SSIM per echo =============")
print(df.pivot(index="echo", columns="radius", values=["SNR","SSIM"])
        .round(3).to_string())
print("\n============= 평균 (6 echo) =====================")
print(df.groupby("radius").mean(numeric_only=True).round(3))

# ─── 그래프 (반경별 평균) ────────────────────────────────────────────────
avg = df.groupby("radius").mean(numeric_only=True).reset_index()

fig, ax1 = plt.subplots(figsize=(6,4))
ax2 = ax1.twinx()
ax1.bar(avg.radius-0.1, avg.SNR,  width=0.2, label="SNR (dB)")
ax2.bar(avg.radius+0.1, avg.SSIM, width=0.2, color="tab:orange", label="SSIM")

ax1.set_xticks(R_LIST); ax1.set_xlabel("Patch radius r")
ax1.set_ylabel("SNR (dB)"); ax2.set_ylabel("SSIM")
ax1.set_title("MP-PCA radius tuning – mean over 6 echoes")
fig.legend(loc="upper left", bbox_to_anchor=(0.12,0.92))
fig.tight_layout()
plt.savefig("radius_comparison.png", dpi=300)
plt.close()
print("\n✓ 그래프 저장 → radius_comparison.png")