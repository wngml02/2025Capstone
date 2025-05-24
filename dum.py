from pathlib import Path, PurePath

NOISY_MAT = "noisy_meas_gre_dir1_20.mat"
p = Path(NOISY_MAT)
print("Exists:", p.exists(), "size:", p.stat().st_size, "bytes")