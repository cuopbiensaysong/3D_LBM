import os

from datetime import datetime

import torch
import yaml
from monai.utils import set_determinism
# from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from tqdm import tqdm

from generative.losses import PatchAdversarialLoss, PerceptualLoss

from data_3D_loader import get_vqgan_3D_dataloader
from backbone import get_autoencoder_model, get_discriminator_model, get_unet_model_from_ckpt
from utils import setup_wandb, save_config

# for reproducibility purposes set a seed
set_determinism(42)



def save_checkpoint(run_dir: str, filename: str, autoencoder, discriminator, optimizer_g, optimizer_d, epoch: int):
    os.makedirs(run_dir, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "autoencoder_state": autoencoder.state_dict(),
        "discriminator_state": discriminator.state_dict(),
        "optimizer_g_state": optimizer_g.state_dict(),
        "optimizer_d_state": optimizer_d.state_dict(),
    }
    torch.save(checkpoint, os.path.join(run_dir, filename))


def main(
    train_txt_path: str = '/home/huutien/5_folds_split_3D/VQGAN/fold_1_train.txt',
    val_txt_path: str = '/home/huutien/5_folds_split_3D/VQGAN/fold_1_val.txt',
    data_dir: str = '/home/huutien/filter_ds',
    model_dir: str = '/home/huutien/sources/GenerativeModels/large_files',
    project_name: str = "3D_KL_training_KL",
    run_name="3D_KL_training_KL",
    save_dir="/home/huutien/3D_training/results",
    device='cuda',
    adv_weight=0.01,
    perceptual_weight=0.001,
    kl_weight=1e-6,
    n_epochs=100,
    autoencoder_warm_up_n_epochs=5,
    val_interval=2,  # epochs
    batch_size=4,
    num_workers=4,
    cache_rate=1.0,
    use_bfloat16: bool = True,
    optimizer_g: str = "AdamW",
    optimizer_d: str = "AdamW",
    optimizer_g_params: dict = {
        "lr": 5e-5
    },
    optimizer_d_params: dict = {
        "lr": 5e-5
    },
):
    run_name = run_name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, run_name)

    config = {
        "train_txt_path": train_txt_path,
        "val_txt_path": val_txt_path,
        "data_dir": data_dir,
        "model_dir": model_dir,
        "project_name": project_name,
        "run_name": run_name,
        "save_dir": save_dir,
        "device": device,
        "adv_weight": adv_weight,
        "perceptual_weight": perceptual_weight,
        "kl_weight": kl_weight,
        "n_epochs": n_epochs,
        "autoencoder_warm_up_n_epochs": autoencoder_warm_up_n_epochs,
        "val_interval": val_interval,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "cache_rate": cache_rate,
        "use_bfloat16": use_bfloat16,
    }
    save_config(config, run_dir)

    wandb_run = setup_wandb(project_name, run_name, config)

    train_loader = get_vqgan_3D_dataloader(data_dir, train_txt_path, stage="train", batch_size=batch_size, num_workers=num_workers, cache_rate=cache_rate)
    val_loader = get_vqgan_3D_dataloader(data_dir, val_txt_path, stage="val", batch_size=batch_size, num_workers=num_workers, cache_rate=cache_rate)

    autoencoder = get_autoencoder_model(model_dir)
    autoencoder.to(device)

    discriminator = get_discriminator_model()
    discriminator.to(device)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    def KL_loss(z_mu, z_sigma):
        # Sum over latent dims, then mean over batch to avoid shape errors.
        kl_loss = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
            dim=[1, 2, 3, 4],
        )
        return kl_loss.mean()
    
    optimizer_g = getattr(torch.optim, optimizer_g)
    optimizer_d = getattr(torch.optim, optimizer_d)
    optimizer_g = optimizer_g(params=autoencoder.parameters(), **optimizer_g_params)
    optimizer_d = optimizer_d(params=discriminator.parameters(), **optimizer_d_params)

    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_loss_list = []

    best_val_recon_loss = float("inf")
    device_type = device.split(":")[0]
    amp_enabled = use_bfloat16 and device_type == "cuda" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    if amp_enabled:
        print("Using bfloat16 autocast for training.")
    else:
        print("Autocast disabled; training in float32.")

    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0.0
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0
        gen_steps = 0
        disc_steps = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            images = images.to(device)  # choose only one of Brats channels

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            generator_loss = None
            discriminator_loss = None
            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                reconstruction, z_mu, z_sigma = autoencoder(images)
                kl_loss = KL_loss(z_mu, z_sigma)

                recons_loss = l1_loss(reconstruction, images)
                p_loss = loss_perceptual(reconstruction, images)
                kl_term = kl_weight * kl_loss

                p_term = perceptual_weight * p_loss
                loss_g = recons_loss + kl_term + p_term

                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if generator_loss is not None and discriminator_loss is not None:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()
                gen_steps += 1
                disc_steps += 1

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / max(1, gen_steps),
                    "disc_loss": disc_epoch_loss / max(1, disc_steps),
                }
            )

            if wandb_run:
                wandb_run.log(
                    {
                        "train/recons_loss": recons_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/perceptual_loss": p_loss.item(),
                        "train/generator_loss": generator_loss.item() if generator_loss is not None else 0.0,
                        "train/discriminator_loss": discriminator_loss.item() if discriminator_loss is not None else 0.0,
                        "epoch": epoch,
                        "step": epoch * len(train_loader) + step,
                    }
                )

        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / max(1, gen_steps))
        epoch_disc_loss_list.append(disc_epoch_loss / max(1, disc_steps))

        if wandb_run:
            wandb_run.log(
                {
                    "epoch/train_recons_loss": epoch_recon_loss_list[-1],
                    "epoch/train_gen_loss": epoch_gen_loss_list[-1],
                    "epoch/train_disc_loss": epoch_disc_loss_list[-1],
                    "epoch": epoch,
                }
            )

        # Validation
        if (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_recon_loss = 0.0
            val_kl_loss_total = 0.0
            val_p_loss_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_images = val_batch["image"]
                    if hasattr(val_images, "as_tensor"):
                        val_images = val_images.as_tensor()
                    val_images = val_images.to(device)
                    with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                        val_reconstruction, val_z_mu, val_z_sigma = autoencoder(val_images)
                        val_kl_loss = KL_loss(val_z_mu, val_z_sigma)
                        val_recons_loss = l1_loss(val_reconstruction, val_images)
                        val_p_loss = loss_perceptual(val_reconstruction, val_images)
                    val_recon_loss += val_recons_loss.item()
                    val_kl_loss_total += val_kl_loss.item()
                    val_p_loss_total += val_p_loss.item()
                    val_steps += 1

            mean_val_recon = val_recon_loss / max(1, val_steps)
            mean_val_kl = val_kl_loss_total / max(1, val_steps)
            mean_val_p = val_p_loss_total / max(1, val_steps)
            val_recon_loss_list.append(mean_val_recon)

            if wandb_run:
                wandb_run.log(
                    {
                        "val/recons_loss": mean_val_recon,
                        "val/kl_loss": mean_val_kl,
                        "val/perceptual_loss": mean_val_p,
                        "epoch": epoch,
                    }
                )

            # Save best checkpoint based on validation reconstruction loss
            if mean_val_recon < best_val_recon_loss:
                best_val_recon_loss = mean_val_recon
                save_checkpoint(run_dir, "best_checkpoint.pth", autoencoder, discriminator, optimizer_g, optimizer_d, epoch)

        # Save last checkpoint every epoch
        save_checkpoint(run_dir, "last_checkpoint.pth", autoencoder, discriminator, optimizer_g, optimizer_d, epoch)

    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()


if __name__ == "__main__":
    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(**config)