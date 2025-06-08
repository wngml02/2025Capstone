import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ─── 데이터 로드 ────────────────────────────────────────────────────────────
mat = sio.loadmat("meas_gre_dir1.mat", simplify_cells=True)
orig_cplx = mat["meas_gre"].astype(np.complex64)
mask = mat["mask_brain"].astype(bool)

# ─── 파라미터 설정 ─────────────────────────────────────────────────────────
echo = 0
z_idx = 88
real = np.real(orig_cplx[:, :, :, echo])
imag = np.imag(orig_cplx[:, :, :, echo])

# ─── 시각화 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle(f"Original Real / Imag / Mask (Echo {echo}, Slice {z_idx})", fontsize=30)

axes[0].imshow(real[:, :, z_idx], cmap='gray')
axes[0].set_title("Real", fontsize=24)
axes[0].axis("off")

axes[1].imshow(imag[:, :, z_idx], cmap='gray')
axes[1].set_title("Imaginary", fontsize=24)
axes[1].axis("off")

axes[2].imshow(mask[:, :, z_idx], cmap='gray')
axes[2].set_title("Mask", fontsize=24)
axes[2].axis("off")

plt.subplots_adjust(wspace=0.05, hspace=0, top=0.90)

# ─── 저장 (선택) ────────────────────────────────────────────────────────────
out_path = f"orig_echo{echo}_slice{z_idx:03d}_real_imag_mask.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"✔ 저장 완료 → {out_path}")

plt.show()
