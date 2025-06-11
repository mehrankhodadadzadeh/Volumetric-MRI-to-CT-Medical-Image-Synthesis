# trainer.py ─────────────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from monai.metrics import SSIMMetric


def relativistic_average_loss(real_logits, fake_logits):
    real = torch.mean(F.softplus(-(real_logits - fake_logits.mean())))
    fake = torch.mean(F.softplus(-(fake_logits - real_logits.mean())))
    return 0.5 * (real + fake)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GANTrainer:
    """
    Generic 3-D GAN trainer (patch discriminator, L1 + optional GAN loss).

    The first tensor in every batch is treated as *input*,
    the second as *target*. Works for CT→MRI or MRI→CT,
    depending on how the dataloader provides it.
    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        train_loader,
        val_loader,
        device,
        lr: float,
        beta1: float,
        beta2: float,
        checkpoint_dir: str,
        use_gan: bool = True,
    ):
        self.gen     = generator.to(device)
        self.disc    = discriminator.to(device)
        self.use_gan = use_gan
        self.tl      = train_loader
        self.vl      = val_loader
        self.dev     = device
        self.ckpt    = checkpoint_dir

        self.opt_g = optim.Adam(self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_d = optim.Adam(self.disc.parameters(), lr=lr, betas=(beta1, beta2))
        self.l1    = nn.L1Loss()
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_g, mode="min", factor=0.5, patience=100, verbose=True
        )

        self.ssim = SSIMMetric(data_range=1.0, spatial_dims=3, reduction="mean")

        os.makedirs(self.ckpt, exist_ok=True)
        wandb.log({
            "model/generator_params":     count_parameters(self.gen),
            "model/discriminator_params": count_parameters(self.disc),
        })

        self._train_losses_g = []
        self._train_losses_d = []
        self._val_losses     = []
        self._epochs         = []

    def train(self, epochs: int, return_metrics: bool = False):
        best_val  = float("inf")
        best_mae  = best_psnr = best_ssim = None

        for ep in range(1, epochs + 1):
            self.gen.train()
            if self.use_gan:
                self.disc.train()
            g_losses, d_losses = [], []

            for src, tgt, _ in tqdm(self.tl, desc=f"Epoch {ep} [Train]"):
                src, tgt = src.to(self.dev), tgt.to(self.dev)
                fake = self.gen(src)

                # Discriminator step (only if GAN enabled)
                if self.use_gan:
                    d_loss = relativistic_average_loss(
                        self.disc(tgt), self.disc(fake.detach())
                    )
                    self.opt_d.zero_grad()
                    d_loss.backward()
                    self.opt_d.step()
                    d_losses.append(d_loss.item())

                # Generator step
                l1 = self.l1(fake, tgt)
                if self.use_gan:
                    gan    = relativistic_average_loss(self.disc(fake), self.disc(tgt))
                    g_loss = gan + 10 * l1
                else:
                    g_loss = 10 * l1

                self.opt_g.zero_grad()
                g_loss.backward()
                self.opt_g.step()
                g_losses.append(g_loss.item())

            # Validation
            val_loss, mae, psnr, ssim = self._validate()
            self.sched.step(val_loss)

            # Save best
            if val_loss < best_val:
                best_val, best_mae, best_psnr, best_ssim = val_loss, mae, psnr, ssim
                torch.save(self.gen.state_dict(),
                           os.path.join(self.ckpt, "best_generator.pth"))

            if ep % 500 == 0:
                torch.save(self.gen.state_dict(),
                           os.path.join(self.ckpt, f"generator_epoch_{ep}.pth"))

            avg_g = float(np.mean(g_losses))
            avg_d = float(np.mean(d_losses)) if self.use_gan else None

            log_dict = {
                "loss/train_gen": avg_g,
                "loss/val":       val_loss,
                "metrics/mae":    mae,
                "metrics/psnr":   psnr,
                "metrics/ssim":   ssim,
                "epoch":          ep,
                "lr":             self.opt_g.param_groups[0]["lr"],
            }
            if self.use_gan:
                log_dict["loss/train_disc"] = avg_d

            wandb.log(log_dict)

            self._train_losses_g.append(avg_g)
            if self.use_gan:
                self._train_losses_d.append(avg_d)
            self._val_losses.append(val_loss)
            self._epochs.append(ep)

            disc_str = f" | train_d={avg_d:.4f}" if self.use_gan else ""
            print(
                f"Epoch {ep}: train_g={avg_g:.4f}{disc_str} | "
                f"val={val_loss:.4f} | MAE={mae:.4f} | PSNR={psnr:.2f} | SSIM={ssim:.4f}"
            )

        if return_metrics:
            return best_mae, best_psnr, best_ssim

    def _validate(self):
        self.gen.eval()
        if self.use_gan:
            self.disc.eval()
        self.ssim.reset()

        losses, maes, psnrs, ssims = [], [], [], []
        with torch.no_grad():
            for src, tgt, mask in tqdm(self.vl, desc="[Validation]"):
                src, tgt = src.to(self.dev), tgt.to(self.dev)
                mask_b   = (mask > 0.5).to(self.dev)

                pred = self.gen(src)
                l1   = self.l1(pred, tgt)
                if self.use_gan:
                    gan = relativistic_average_loss(self.disc(pred), self.disc(tgt))
                    losses.append((gan + 10 * l1).item())
                else:
                    losses.append((10 * l1).item())

                # compute metrics inside mask
                p = pred.squeeze(1).cpu().numpy()
                g = tgt.squeeze(1).cpu().numpy()
                m = mask_b.squeeze(1).cpu().numpy().astype(bool)
                if not m.any():
                    continue

                p[~m], g[~m] = 0, 0
                flat_p, flat_g = p[m], g[m]

                maes.append(float(np.mean(np.abs(flat_p - flat_g))))
                mse = float(np.mean((flat_p - flat_g) ** 2))
                dr  = float(flat_g.max() - flat_g.min())
                if dr >= 1e-8:
                    psnrs.append(10 * np.log10((dr**2) / (mse + 1e-12)))

                ssim_val = self.ssim(pred * mask_b, tgt * mask_b)
                if ssim_val.numel() > 1:
                    ssim_val = ssim_val.mean()
                ssims.append(ssim_val.item())

        return (
            float(np.mean(losses)),
            float(np.nanmean(maes)),
            float(np.nanmean(psnrs)),
            float(np.nanmean(ssims)),
        )
