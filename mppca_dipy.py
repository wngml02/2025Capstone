
from pathlib import Path

import napari
import numpy as np
import scipy.io as sio
from dipy.denoise.localpca import localpca, mppca
from imageio import imwrite

# ─── USER PATHS ─────────────────────────────────────────────────────────────
ORIG_MAT   = "meas_gre_dir1.mat"
NOISY_MAT  = "noisy_meas_gre_dir1.mat"
OUT_MAT    = "denoised_real_imag.mat"
SS_DIR     = Path("screens");  SS_DIR.mkdir(exist_ok=True)
PATCH_R    = 2    # (2 → 5×5×5 patch)

# ─── LOAD ORIGINAL & MASK ───────────────────────────────────────────────────
print("⋯ loading original complex GRE")
m0            = sio.loadmat(ORIG_MAT,  simplify_cells=True)
orig_complex  = m0["meas_gre"].astype(np.complex64)          # (256,224,176,6)
mask          = m0["mask_brain"].astype(bool)

print("⋯ loading noisy real / imag")
m1           = sio.loadmat(NOISY_MAT, simplify_cells=True)
noisy_real   = m1["noisy_real"].astype(np.float32)
noisy_imag   = m1["noisy_imag"].astype(np.float32)

assert noisy_real.shape == orig_complex.shape, "shape mismatch!"

# ─── HELPER : MP-PCA with sign preservation ────────────────────────────────
def denoise_preserve_sign(vol4d: np.ndarray) -> np.ndarray:
    """LocalPCA(|vol|) + restore original sign (works on DIPY ≤1.7)."""
    sign   = np.sign(vol4d)          # (-1, 0, +1)
    absvol = np.abs(vol4d)

    # 1. σ 추정 (4-D → 4-D) → 2. echo 차원 평균해 3-D sigma
    sigma4d = mppca(absvol, mask=mask, patch_radius=PATCH_R)
    sigma   = sigma4d.mean(axis=3).astype(np.float32)  # (X,Y,Z)

    den_abs = localpca(absvol, sigma=sigma, mask=mask,
                       patch_radius=PATCH_R).astype(np.float32)
    return den_abs * sign            # 부호 복원

# ─── DENOISE REAL / IMAG (4-D STACK) ───────────────────────────────────────
print("⋯ MP-PCA denoising REAL  (sign-safe)")
den_real = denoise_preserve_sign(noisy_real)

print("⋯ MP-PCA denoising IMAG  (sign-safe)")
den_imag = denoise_preserve_sign(noisy_imag)

sio.savemat(OUT_MAT, {"den_real": den_real, "den_imag": den_imag})
print(f"✔ denoised volumes saved  →  {OUT_MAT}")

# ─── QA MAGNITUDES & FIXED DISPLAY RANGE ───────────────────────────────────
mag_orig  = np.abs(orig_complex)
mag_noisy = np.abs(noisy_real + 1j*noisy_imag)
mag_den   = np.abs(den_real   + 1j*den_imag)

vmin, vmax = np.percentile(mag_orig[mask], [1, 99])
mid_z      = mag_orig.shape[2] // 2

# ─── SAVE 18 PNGs (orig / noisy / den  ×  6 echo) ─────────────────────────
viewer = napari.Viewer(title="MP-PCA QA", show=False)

def snap(volume, tag, echo):
    viewer.add_image(volume[:, :, mid_z, echo], name=tag,
                     contrast_limits=(vmin, vmax), colormap="gray")
    imwrite(SS_DIR / f"echo{echo+1}_{tag}.png",
            viewer.screenshot(canvas_only=True))
    viewer.layers.clear()

print("⋯ capturing screenshots")
for e in range(6):
    snap(mag_orig,  "orig",  e)
    snap(mag_noisy, "noisy", e)
    snap(mag_den,   "den",   e)

print(f"✔ screenshots stored in  {SS_DIR.resolve()}")

# ─── (OPTIONAL) INTERACTIVE napari VIEW ─────────────────────────────────────
for e in range(6):
    viewer.add_image(mag_orig[:,  :, mid_z, e], name=f"O{e+1}",
                     contrast_limits=(vmin, vmax))
    viewer.add_image(mag_noisy[:, :, mid_z, e], name=f"N{e+1}",
                     contrast_limits=(vmin, vmax))
    viewer.add_image(mag_den[:,   :, mid_z, e], name=f"D{e+1}",
                     contrast_limits=(vmin, vmax))

viewer.grid.enabled = True
viewer.grid.shape   = (3, 6)   # rows = O / N / D   ×   cols = echo1-6
napari.run()