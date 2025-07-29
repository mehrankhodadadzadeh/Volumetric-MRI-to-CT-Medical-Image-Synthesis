import os, numpy as np, nibabel as nib, torch
import pandas as pd
from tqdm import tqdm
from models import build_generator
from utils import mae_psnr_ssim

MODEL_PATH  = "/scratch/brussel/112/vsc11217/image_generation/MR_CT/Best_model/generator_epoch_2500.pth"
TEST_DIR    = "/data/brussel/112/vsc11217/3D_UNet_Data/full_data_150_15_15_brain/Brain_data/test"
OUTPUT_DIR  = "/scratch/brussel/112/vsc11217/image_generation/MR_CT/Best_model"
GENERATOR   = "attention_unet"
PATCH_SIZE  = (64, 64, 64)
BASE_CH     = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────── #

os.makedirs(OUTPUT_DIR, exist_ok=True)

def pad_to_mult(a, m):
    d, h, w = a.shape
    pd, ph, pw = [(m - s % m) % m for s in (d, h, w)]
    return np.pad(a, ((0, pd), (0, ph), (0, pw))), (d, h, w)

def unpad(a, orig): D, H, W = orig; return a[:D, :H, :W]

def split(vol, psz):
    PD, PH, PW = psz
    for z in range(0, vol.shape[0], PD):
        for y in range(0, vol.shape[1], PH):
            for x in range(0, vol.shape[2], PW):
                yield (z, y, x), vol[z:z+PD, y:y+PH, x:x+PW]

def merge(chunks, shp, psz):
    out = np.zeros(shp, np.float32)
    cnt = np.zeros(shp, np.float32)
    PD, PH, PW = psz
    for (z, y, x), p in chunks:
        out[z:z+PD, y:y+PH, x:x+PW] += p
        cnt[z:z+PD, y:y+PH, x:x+PW] += 1
    return out / np.maximum(cnt, 1)

# ---------------- build generator -------------------------- #
g_kw = dict(in_channels=1, out_channels=1)
if GENERATOR == "unet3d":
    g_kw["base_channels"] = BASE_CH
elif GENERATOR == "swin_unetr":
    g_kw["img_size"] = PATCH_SIZE

net = build_generator(GENERATOR, **g_kw).to(DEVICE)
net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
net.eval()

# ---------------- inference loop -------------------------- #
metrics_per_patient = []
patients = [p for p in sorted(os.listdir(TEST_DIR)) if os.path.isdir(os.path.join(TEST_DIR, p))]

for pid in tqdm(patients, desc="Patients"):
    p_dir = os.path.join(TEST_DIR, pid)

    mri_nii  = nib.load(os.path.join(p_dir, "mr.nii.gz"))
    ct_nii   = nib.load(os.path.join(p_dir, "ct.nii.gz"))
    mask_nii = nib.load(os.path.join(p_dir, "mask.nii.gz"))

    mri   = mri_nii.get_fdata().astype(np.float32)
    ct    = ct_nii.get_fdata().astype(np.float32)
    mask  = mask_nii.get_fdata().astype(bool)
    affine = mri_nii.affine

    # Normalize MRI input (z-score)
    mri_z = (mri - mri.mean()) / (mri.std() + 1e-8)

    # Patch-based inference
    mri_pad, orig_shape = pad_to_mult(mri_z, PATCH_SIZE[0])
    patches = []
    for pos, patch in split(mri_pad, PATCH_SIZE):
        with torch.no_grad():
            pred = net(torch.from_numpy(patch[None, None]).to(DEVICE))
        patches.append((pos, pred.cpu().squeeze().numpy()))

    ct_pred_norm = unpad(merge(patches, mri_pad.shape, PATCH_SIZE), orig_shape)
    ct_pred_hu   = ct_pred_norm * 3000.0 - 1000.0
    ct_pred_norm = np.clip(ct_pred_norm, 0.0, 1.0)

    # Save HU output for clinical use
    nib.save(nib.Nifti1Image(ct_pred_hu.astype(np.float32), affine),
             os.path.join(OUTPUT_DIR, f"synth_ct_hu_{pid}.nii.gz"))

    # Normalize GT CT for evaluation
    ct_gt_clip = np.clip(ct, -1000, 2000)
    ct_gt_norm = (ct_gt_clip + 1000) / 3000.0

    # Evaluate on normalized scale
    mae, psnr, ssim = mae_psnr_ssim(ct_pred_norm, ct_gt_norm, mask)
    print(f"{pid:>8}  MAE={mae:.4f} | PSNR={psnr:.2f} dB | SSIM={ssim:.4f}")
    metrics_per_patient.append({
        "PatientID": pid,
        "MAE": round(mae, 4),
        "PSNR": round(psnr, 2),
        "SSIM": round(ssim, 4),
    })

# ---------------- Save Summary CSV -------------------------- #
df = pd.DataFrame(metrics_per_patient)
mean_row = {
    "PatientID": "MEAN",
    "MAE": round(df["MAE"].mean(), 4),
    "PSNR": round(df["PSNR"].mean(), 2),
    "SSIM": round(df["SSIM"].mean(), 4),
}
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
df.to_csv(csv_path, index=False)
print(f"\n📄 Metrics saved to: {csv_path}")

# ---------------- Final Console Summary -------------------------- #
print("\nFINAL → MAE %.4f | PSNR %.2f dB | SSIM %.4f"
      % (mean_row["MAE"], mean_row["PSNR"], mean_row["SSIM"]))
