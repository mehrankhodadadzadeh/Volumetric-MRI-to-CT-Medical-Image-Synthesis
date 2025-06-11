import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from monai.transforms import (
    Compose,
    RandSpatialCropd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandAffined,
    RandScaleIntensityd,
    RandGaussianNoised,
    ToTensord,
)


class PairedNiftiDataset(Dataset):
    """Load paired **MRI → CT** volumes (with binary mask) and apply identical
    spatial/intensity augmentations to *image* (MRI), *label* (CT) and *mask*.

    ─ image  : generator **input**  (normalized MRI)
    ─ label  : ground‑truth **target** (windowed CT in [0, 1])
    ─ mask   : foreground / evaluation mask (unchanged)
    """

    def __init__(
        self,
        root_dir: str,
        patch_size: tuple = (64, 64, 64),
        mode: str = "train",
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode.lower()
        self.augment = augment
        self.patient_dirs = sorted(glob(os.path.join(root_dir, "*")))

        # ───────────────────────── transforms ──────────────────────────
        if self.augment and self.mode == "train":
            self.transform = Compose(
                [
                    # spatial ops applied to image, label, mask
                    RandSpatialCropd(
                        keys=["image", "label", "mask"],
                        roi_size=patch_size,
                        random_size=False,
                    ),
                    RandRotated(
                        keys=["image", "label", "mask"],
                        range_x=10,
                        range_y=10,
                        range_z=0,
                        prob=0.3,
                        mode=("bilinear", "nearest", "nearest"),
                        padding_mode="zeros",
                    ),
                    RandFlipd(
                        keys=["image", "label", "mask"],
                        spatial_axis=0,
                        prob=0.2,
                    ),
                    RandZoomd(
                        keys=["image", "label", "mask"],
                        min_zoom=0.9,
                        max_zoom=1.1,
                        prob=0.2,
                        mode=("bilinear", "nearest", "nearest"),
                    ),
                    RandAffined(
                        keys=["image", "label", "mask"],
                        translate_range=(-5, 5),
                        prob=0.2,
                        mode=("bilinear", "nearest", "nearest"),
                        padding_mode="zeros",
                    ),
                    # intensity noise **only on MRI input**
                    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
                    RandGaussianNoised(keys=["image"], std=0.01, prob=0.2),
                    # to PyTorch tensors
                    ToTensord(keys=["image", "label", "mask"]),
                ]
            )
        else:
            # validation/test: deterministic crop + tensor
            self.transform = Compose(
                [
                    RandSpatialCropd(
                        keys=["image", "label", "mask"],
                        roi_size=patch_size,
                        random_size=False,
                    ),
                    ToTensord(keys=["image", "label", "mask"]),
                ]
            )

    # ────────────────────────── helpers ────────────────────────────
    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]

        # ------------------- load NIfTI volumes --------------------
        mri_path = os.path.join(patient_dir, "mr.nii.gz")  # input
        ct_path = os.path.join(patient_dir, "ct.nii.gz")   # target
        mask_path = os.path.join(patient_dir, "mask.nii.gz")

        mri_nii = nib.load(mri_path)
        ct_nii = nib.load(ct_path)
        mask_nii = nib.load(mask_path)

        mri = mri_nii.get_fdata().astype(np.float32)
        ct = ct_nii.get_fdata().astype(np.float32)
        mask = mask_nii.get_fdata().astype(np.float32)

        # ------------------- intensity normalisation --------------
        mri = (mri - mri.mean()) / (mri.std() + 1e-8)  # z‑score now between -4 to 4.
        ct = np.clip(ct, -1000, 2000)
        ct = (ct + 1000) / 3000.0    # between 0 to 1
                            

        # ------------------- reshape to (C, D, H, W) --------------
        mri = np.expand_dims(mri, axis=0)
        ct = np.expand_dims(ct, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {"image": mri, "label": ct, "mask": mask}
        sample = self.transform(sample)

        return sample["image"], sample["label"], sample["mask"]


# ======================================================================
# convenience wrapper for typical train/val/test loaders
# ======================================================================

def get_dataloaders(
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int = 1,
    patch_size: tuple = (64, 64, 64),
):
    train_ds = PairedNiftiDataset(
        train_dir, patch_size=patch_size, mode="train", augment=True
    )
    val_ds = PairedNiftiDataset(val_dir, patch_size=patch_size, mode="val", augment=False)
    test_ds = PairedNiftiDataset(
        test_dir, patch_size=patch_size, mode="test", augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader, test_loader
