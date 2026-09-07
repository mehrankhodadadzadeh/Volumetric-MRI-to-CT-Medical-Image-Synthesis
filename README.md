


Volumetric MRI-to-CT Medical Image Synthesis
End-to-end PyTorch/MONAI framework for paired 3D brain MRI-to-CT synthesis. The project trains volumetric encoder-decoder networks to generate a synthetic CT (sCT) from an aligned MRI volume and evaluates the result within an anatomical mask.

This repository contains the MRI → CT component of my master's thesis on volumetric cross-modality medical image translation. Implementations for additional modality pairs are available in my other public GitHub repositories.

Research use only: The generated images are not validated for clinical diagnosis, treatment planning, or patient care.

Example result

Key features
Fully 3D paired MRI-to-CT translation

Three interchangeable generator architectures:

3D U-Net

3D Attention U-Net

Swin UNETR

L1-only training or optional adversarial training

Memory-efficient patch-based training and whole-volume inference

Paired spatial augmentation using MONAI

Anatomical-mask-based evaluation with MAE, PSNR, and 3D SSIM

Simple train/validation workflow and five-fold cross-validation

Experiment tracking with Weights & Biases

NIfTI output in Hounsfield units and per-patient CSV summaries

Method overview
The input MRI is normalized using z-score normalization. Target CT intensities are clipped to [-1000, 2000] HU and linearly scaled to [0, 1] during training. The selected generator predicts a synthetic CT patch from the corresponding MRI patch.

Training can use either:

L1 reconstruction: 10 × L1

Adversarial + reconstruction: relativistic average adversarial loss + 10 × L1

During inference, each MRI volume is padded, divided into 3D patches, reconstructed, converted back to Hounsfield units, and saved as a NIfTI image. Quantitative evaluation is performed against the registered ground-truth CT.

Repository structure
.
├── main.py       # experiment configuration, training, and cross-validation
├── dataset.py    # paired NIfTI loading, normalization, augmentation, loaders
├── models.py     # 3D U-Net, Attention U-Net, Swin UNETR, discriminator
├── trainer.py    # optimization, validation, checkpointing, W&B logging
├── inference.py  # whole-volume patch inference and result export
├── utils.py      # MAE, PSNR, and 3D SSIM utilities
└── test.png      # example synthesis result
Dataset organization
The code expects separate training, validation, and test directories. Each patient directory must contain a spatially aligned MRI, CT, and binary anatomical mask:

dataset_root/
├── train/
│   ├── patient_001/
│   │   ├── mr.nii.gz
│   │   ├── ct.nii.gz
│   │   └── mask.nii.gz
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
All three volumes for a patient must share the same voxel grid, orientation, and spatial dimensions. Registration and preprocessing are expected to be completed before using this pipeline.

Installation
Python 3.10+ and a CUDA-capable GPU are recommended.

git clone https://github.com/mehrankhodadadzadeh/Volumetric-MRI-to-CT-Medical-Image-Synthesis.git
cd Volumetric-MRI-to-CT-Medical-Image-Synthesis

python -m venv .venv
source .venv/bin/activate
pip install torch monai nibabel numpy pandas scikit-image scikit-learn tqdm wandb
Install the PyTorch build appropriate for your CUDA version using the official PyTorch installation guide.

Configuration
Experiment settings are currently defined in the cfg dictionary in main.py. Before training, update:

train_data_dir = "/path/to/dataset/train"
val_data_dir   = "/path/to/dataset/val"
test_data_dir  = "/path/to/dataset/test"
checkpoint_dir = "/path/to/checkpoints"
results_dir    = "/path/to/results"
Choose a generator, objective, and evaluation mode:

generator = "unet3d"         # or "attention_unet", "swin_unetr"
use_gan   = False             # False: L1 only; True: adversarial + L1
mode      = "simple"          # or "crossval"
Other configurable parameters include patch size, batch size, learning rate, Adam betas, base channels, random seed, and number of epochs.

Authenticate Weights & Biases through the environment or CLI rather than storing a key in source code:

wandb login
Training
After updating the configuration:

python main.py
The best generator is saved as best_generator.pth according to validation loss. Additional checkpoints are saved every 500 epochs.

Inference
Update the paths and model settings at the top of inference.py so that they match the trained checkpoint:

MODEL_PATH = "/path/to/best_generator.pth"
TEST_DIR   = "/path/to/dataset/test"
OUTPUT_DIR = "/path/to/output"
GENERATOR  = "attention_unet"
PATCH_SIZE = (64, 64, 64)
Then run:

python inference.py
For every test subject, the script produces:

synth_ct_hu_<patient_id>.nii.gz — synthetic CT in Hounsfield units

metrics_summary.csv — per-patient and mean MAE, PSNR, and SSIM

Evaluation metrics
MAE: average absolute voxel error

PSNR: peak signal-to-noise ratio between predicted and reference CT

SSIM: 3D structural similarity

Metrics are computed on normalized CT volumes with support for anatomical masking. Because CT is scaled over a 3000 HU window, normalized MAE can be converted approximately to HU as:

MAE_HU ≈ MAE_normalized × 3000
Thesis context
This work was developed as part of an MSc thesis in Biomedical Engineering (Artificial Intelligence and Digital Health) at Ghent University and Vrije Universiteit Brussel (VUB). The broader thesis investigated volumetric cross-modality synthesis across MRI, CT, and FET-PET using convolutional, attention-based, transformer-based, and adversarial learning approaches.

Limitations
The pipeline assumes paired, pre-registered MRI and CT volumes.

Dataset paths and experiment settings are currently configured directly in the scripts.

Pretrained model weights and medical datasets are not included.

Clinical validity and generalization to new scanners, institutions, anatomies, or acquisition protocols have not been established.

The current inference implementation uses non-overlapping patches; boundary artifacts may therefore occur.

Contact
For questions, research discussion, or collaboration, contact Mehran Khodadadzadeh at mehrankhodadadzadeh90@gmail.com.

If this repository supports your work, please cite or link to the repository. A formal citation will be added if the thesis or related research is published.

License
No open-source license is currently included. Unless a license is added, the source code remains under the author's default copyright and may not be redistributed or reused beyond the permissions provided by applicable law. Please contact the author for reuse or collaboration.
