from pathlib import Path

NoiseLvl     = 50                    
PATCH_R      = 3                      # MP-PCA patch radius

ORIG_MAT     = "meas_gre_dir1.mat"               
NOISY_MAT    = f"noisy_meas_gre_dir1_{NoiseLvl}.mat"
OUT_MAT      = f"denoised_real_imag_{NoiseLvl}_sqrt_r{PATCH_R}.mat"
OUT_DIR      = Path(f"dn_split_{NoiseLvl}_r{PATCH_R}")
OUT_DIR.mkdir(exist_ok=True)

EXCEL_FILE   = Path(f"snr_ssim_report_{NoiseLvl}.xlsx")


import numpy as np
import pandas as pd
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim

from localpca_dn import mppca

m0         = sio.loadmat(ORIG_MAT, simplify_cells=True)
orig_cplx  = m0["meas_gre"].astype(np.complex64)            # (X,Y,Z,C)
mask       = m0["mask_brain"].astype(bool)                  # (X,Y,Z)

m1         = sio.loadmat(NOISY_MAT, simplify_cells=True)
noisy_real = m1["noisy_real"].astype(np.float32)
noisy_imag = m1["noisy_imag"].astype(np.float32)

# MP-PCA 디노이즈
def mppca_dn(vol4d, mask):
    return mppca(vol4d, mask=mask, patch_radius=PATCH_R)

print("⋯ MP-PCA denoise REAL")
den_real = mppca_dn(noisy_real, mask)
print("⋯ MP-PCA denoise IMAG")
den_imag = mppca_dn(noisy_imag, mask)
den_cplx = den_real + 1j * den_imag

sio.savemat(OUT_DIR/OUT_MAT, {"den_real": den_real,
                            "den_imag": den_imag,
                            "den_cplx": den_cplx})
print("✔ Denoised .mat saved →", OUT_DIR/OUT_MAT)

# magnitude 계산산
orig_mag  = np.abs(orig_cplx)
noisy_mag = np.sqrt(noisy_real**2 + noisy_imag**2)
deno_mag  = np.sqrt(den_real**2  + den_imag**2)

def snr_diff_rician(ref, test, roi):
    s=[]
    for c in range(ref.shape[3]):
        mu   = ref[...,c][roi].mean()
        sigma_raw = (test[...,c] - ref[...,c])[roi].std(ddof=1) # 아직 미정정
        s_corr = np.sqrt(max(mu**2 - 2*sigma_raw**2, 0))
        s.append(s_corr / sigma_raw)
    return np.asarray(s)

def ssim_echo(ref, test, roi):
    X,Y,Z,C = ref.shape
    vals = np.zeros(C)
    for c in range(C):
        s_list=[]
        for z in range(Z):
            if not roi[:,:,z].any(): continue
            r2,t2 = ref[:,:,z,c], test[:,:,z,c]
            dr = r2.max() - r2.min()
            if dr==0: continue
            s_val,_ = ssim(r2, t2, data_range=dr, full=True)
            s_list.append(s_val)
        vals[c] = np.mean(s_list)
    return vals


snr_before = snr_diff_rician(orig_mag, noisy_mag, mask)
snr_after  = snr_diff_rician(orig_mag, deno_mag,  mask)

ssim_before = ssim_echo(orig_mag, noisy_mag, mask)
ssim_after  = ssim_echo(orig_mag, deno_mag,  mask)

df_echo = pd.DataFrame({
    "Echo":          np.arange(len(snr_before)),
    "SNR_before":    snr_before,
    "SNR_after":     snr_after,
    "ΔSNR":          snr_after - snr_before,
    "SSIM_before":   ssim_before,
    "SSIM_after":    ssim_after,
    "ΔSSIM":         ssim_after - ssim_before
}).round(3)


avg = df_echo.drop(columns="Echo").mean().to_dict()
avg["Echo"]="Average"
df_echo = pd.concat([df_echo, pd.DataFrame([avg])], ignore_index=True)

print("\nEcho-wise Performance Comparison (Before vs After Denoising):")
pd.options.display.float_format = "{:,.3f}".format
print(df_echo.to_string(index=False))


def snr_slice_df(ref, test, roi, label):
    rows=[]
    X,Y,Z,C=ref.shape
    for z in range(Z):
        m2d=roi[:,:,z]
        if not m2d.any(): continue
        for c in range(C):
            mu   = ref[:,:,z,c][m2d].mean()
            sigma=(test[:,:,z,c]-ref[:,:,z,c])[m2d].std(ddof=1)/np.sqrt(2)
            s_corr=np.sqrt(max(mu**2-2*sigma**2,0))
            rows.append({"Slice":z,"Echo":c,"{}".format(label):s_corr/sigma})
    return pd.DataFrame(rows)

def ssim_slice_df(ref, test, roi, label):
    rows=[]
    X,Y,Z,C=ref.shape
    for z in range(Z):
        if not roi[:,:,z].any(): continue
        for c in range(C):
            r2,t2 = ref[:,:,z,c], test[:,:,z,c]
            dr = r2.max()-r2.min()
            if dr==0: continue
            val,_ = ssim(r2, t2, data_range=dr, full=True)
            rows.append({"Slice":z,"Echo":c,label:val})
    return pd.DataFrame(rows)

df_snr_b = snr_slice_df(orig_mag,noisy_mag,mask,"SNR_before")
df_snr_a = snr_slice_df(orig_mag,deno_mag ,mask,"SNR_after")
df_ssim_b= ssim_slice_df(orig_mag,noisy_mag,mask,"SSIM_before")
df_ssim_a= ssim_slice_df(orig_mag,deno_mag ,mask,"SSIM_after")

df_slice = (df_snr_b.merge(df_snr_a,  on=["Slice","Echo"])
                    .merge(df_ssim_b, on=["Slice","Echo"])
                    .merge(df_ssim_a, on=["Slice","Echo"])
                    .assign(ΔSNR  = lambda d:d["SNR_after"] - d["SNR_before"],
                            ΔSSIM = lambda d:d["SSIM_after"]- d["SSIM_before"])
                    .loc[:,["Echo","Slice",
                            "SNR_before","SNR_after","ΔSNR",
                            "SSIM_before","SSIM_after","ΔSSIM"]]
                    .round(3)
                    .sort_values(["Echo","Slice"],ignore_index=True))

# Excel 저장
with pd.ExcelWriter(EXCEL_FILE, engine="xlsxwriter") as xw:
    df_echo .to_excel(xw, sheet_name="EchoSummary",  index=False)
    df_slice.to_excel(xw, sheet_name="SliceDetail", index=False)

print("\n✔ Excel saved →", EXCEL_FILE.resolve())