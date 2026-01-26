import argparse
import os
import omegaconf
import torch
import nibabel as nib
from generative.networks.nets import DiffusionModelUNet
from backbone import get_autoencoder_model_from_ckpt
from ema import EMA
from tqdm import tqdm
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
import pandas as pd

from data_3D_loader import get_i2i_3D_dataloader_for_inference


parser = argparse.ArgumentParser()
parser.add_argument("--train_csv_path", type=str, default='./data/fold_1_train_balanced.csv')
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=20)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--num_outputs_per_sample", type=int, default=1)


args = parser.parse_args()

class Inference():
    def __init__(self, args): 
        self.args = args

        self.num_outputs_per_sample = args.num_outputs_per_sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_csv_path = args.train_csv_path

        self.output_dir = args.output_dir
        os.makedirs(os.path.join(self.output_dir, "AD"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "CN"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "MCI"), exist_ok=True)
        

        self.num_inference_steps = args.num_inference_steps
        self.ckpt_path = args.ckpt_path
        self.cfg = omegaconf.OmegaConf.load(args.config_path)
        self.use_ema = self.cfg.EMA.use_ema
        self.load_unet()
        
        # Load VAE if it exists in config, otherwise None
        if hasattr(self.cfg, 'vae_path') and self.cfg.vae_path:
            self.vae = get_autoencoder_model_from_ckpt(self.cfg.vae_path)
            self.vae.to(self.device, dtype=torch.float32)
        else:
            self.vae = None

        self.sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
        )

        self.dtype = torch.float32
        self.unet.to(self.device, dtype=self.dtype)
        self.bridge_noise_sigma = self.cfg.lbm.bridge_noise_sigma

        self.test_data_loader = get_i2i_3D_dataloader_for_inference(self.train_csv_path, root_dir=self.cfg.data_dir, batch_size=1)

        self.unet.eval()
        if self.use_ema:
            self.ema.apply_shadow(self.unet)

        # --- Initialize Metrics ---

    def load_unet(self):
        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        unet_model = DiffusionModelUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=3,
            num_channels=(256, 512, 768),
            num_res_blocks=2,
            attention_levels=(False, True, True),
            norm_num_groups=32,
            norm_eps=1e-6,
            resblock_updown=True,
            num_head_channels=[0, 512, 768],
            with_conditioning=True,
            transformer_num_layers=1,
            cross_attention_dim=4,
            upcast_attention=True,
            use_flash_attention=False,
        )
        unet_model.load_state_dict(ckpt["unet_state"], strict=True)
        unet_model.to(self.device)
        unet_model.eval()
        print(f"Loaded UNet model from {self.ckpt_path} at epoch {ckpt['epoch']} on {self.device}")
        self.unet = unet_model
        if self.use_ema:
            self.ema = EMA(self.cfg.EMA.ema_decay)
            self.ema.shadow = ckpt['ema_state']
            self.ema.reset_device(self.unet)
        
        return unet_model



    def infer(self):
        print(f"Starting inference on {len(self.test_data_loader)} samples...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_data_loader)):    
                src_img = batch['A'].to(self.device, dtype=self.dtype)
                
                # Encode
                if self.vae is not None:
                    z = self.vae.encode_stage_2_inputs(src_img)
                else:
                    z = src_img
            
                # Generate
                # We assume batch_size=1 here based on loader
                for j in range(self.num_outputs_per_sample):
                    decoded_sample = self.sample(z, num_steps=self.num_inference_steps)
                    # Save Output
                    self.save_output(batch, decoded_sample, i, j)


    # def save_output(self, batch, decoded_sample, i, j): # batch_size = 4 
    #     # TODO: save the NIfTI file 
    #     for sample in batch:
    #         sample_label, sample_id = sample['label'], sample['subject_id']
    #         save_path = os.path.join(self.output_dir, f"{sample_label}", f"{sample_id}_{i}_{j}.nii.gz")
    #         pass 
    def save_output(self, batch, decoded_sample, batch_idx, sample_idx):
        """
        Saves the inferred 3D volume to disk as a NIfTI file.
        
        Args:
            batch: The input batch dictionary from the DataLoader.
            decoded_sample: The output tensor from the model (B, C, D, H, W).
            batch_idx: The index of the current batch (for unique naming).
            sample_idx: The index of the generation per sample (if generating multiple per input).
        """
        # 1. Detach from GPU, move to CPU, convert to Numpy
        # Shape is (B, C, D, H, W)
        output_data = decoded_sample.detach().cpu().numpy()
        
        batch_size = output_data.shape[0]

        # 2. Iterate through items in the batch
        for k in range(batch_size):
            # --- A. Retrieve Metadata ---
            # DataLoader collates strings into tuples/lists
            label = batch['label'][k] 
            subj_id = batch['subject_id'][k]
            
            # --- B. Construct Filename ---
            # Structure: output_dir/Label/SubjectID_Batch_Sample.nii.gz
            # Folder created in __init__, but good to ensure specific label folder exists
            save_dir = os.path.join(self.output_dir, str(label))
            
            filename = f"{subj_id}_b{batch_idx}_s{sample_idx}.nii.gz"
            filepath = os.path.join(save_dir, filename)

            # --- C. Process Image Array ---
            # Get the k-th image from the batch
            img_array = output_data[k] # Shape: (C, D, H, W)
            
            # OPTIONAL: Inverse Normalize 
            # (Matches the "x - 0.8" in your dataset.py)
            img_array = img_array + 0.8 

            # Handle Channels for NIfTI
            # If Grayscale (C=1), squeeze to (D, H, W)
            if img_array.shape[0] == 1:
                img_array = img_array.squeeze(0)
            else:
                # If Multi-channel, move Channel to last dim: (D, H, W, C)
                img_array = np.moveaxis(img_array, 0, -1)

            # --- D. Retrieve Affine Matrix ---
            # Crucial for medical images to preserve spatial position.
            # MONAI stores metadata in key_meta_dict. We use input 'A' as reference.
            if 'A_meta_dict' in batch and 'affine' in batch['A_meta_dict']:
                affine = batch['A_meta_dict']['affine'][k].numpy()
            else:
                # Fallback to Identity if metadata is lost
                print(f"Warning: No affine found for {subj_id}, using Identity.")
                affine = np.eye(4)

            # --- E. Save ---
            nifti_img = nib.Nifti1Image(img_array, affine)
            nib.save(nifti_img, filepath)
            
            # Optional: Print status
            # print(f"Saved: {filepath}")

    def _get_conditioning(self, z: torch.Tensor):
        batch_size = z.shape[0]
        conditioning = torch.zeros(batch_size, 1, 4).to(self.device, dtype=self.dtype)
        return conditioning

    def _get_sigmas(self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"):
        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def sample(self, z: torch.Tensor, num_steps: int = 4, verbose: bool = False):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )

        sample = z
        conditioning = self._get_conditioning(z)

        for i, t in enumerate(self.sampling_noise_scheduler.timesteps):
            denoiser_input = sample
            pred = self.unet(
                denoiser_input,
                timesteps=t.to(z.device).repeat(denoiser_input.shape[0]),
                context=conditioning,
            )

            sample = self.sampling_noise_scheduler.step(
                pred, t, sample, return_dict=False
            )[0]
            
            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep = self.sampling_noise_scheduler.timesteps[i + 1].to(z.device).repeat(sample.shape[0])
                sigmas = self._get_sigmas(
                    self.sampling_noise_scheduler, timestep, n_dim=len(z.shape), device=z.device
                )
                sample = sample + self.bridge_noise_sigma * (sigmas * (1.0 - sigmas)) ** 0.5 * torch.randn_like(sample)
                sample = sample.to(z.dtype)

        if self.vae is not None:
            decoded_sample = self.vae.decode_stage_2_outputs(sample)
        else:
            decoded_sample = sample

        return decoded_sample

if __name__ == "__main__":
    inference = Inference(args)
    inference.infer()
    
