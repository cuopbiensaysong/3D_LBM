import argparse
import logging
import os
import omegaconf
import torch
import yaml
from generative.networks.nets import DiffusionModelUNet
from backbone import get_autoencoder_model_from_ckpt
from ema import EMA
from typing import Optional
from tqdm import tqdm
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
import pandas as pd

from data_3D_loader import get_i2i_3D_dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--test_csv_path", type=str, default=None)
parser.add_argument("--npy_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--num_inference_steps", type=int, default=1)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--config_path", type=str, required=False)
parser.add_argument("--save_img_output", type=bool, default=False)
parser.add_argument("--num_outputs_per_sample", type=int, default=1)
parser.add_argument("--compute_metrics", type=bool, default=False)

args = parser.parse_args()

class Inference():
    def __init__(self, args): 
        self.args = args

        self.num_outputs_per_sample = args.num_outputs_per_sample
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_csv_path = args.test_csv_path
        
        self.npy_path = args.npy_path

        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.num_inference_steps = args.num_inference_steps
        self.ckpt_path = args.ckpt_path
        self.save_img_output = args.save_img_output
        if self.save_img_output:
            self.save_img_dir = os.path.join(self.output_dir, "images")
            os.makedirs(self.save_img_dir, exist_ok=True)
        else:
            self.save_img_dir = None
        
        self.cfg = omegaconf.OmegaConf.load(args.config_path)
        self.use_ema = self.cfg.EMA.use_ema
        self.unet = self.load_unet()
        self.vae = get_autoencoder_model_from_ckpt(self.cfg.vae_path)
        self.sampling_noise_scheduler  = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
        )
        # if self.cfg.use_bfloat16:
        #     self.dtype = torch.bfloat16
        # else:
        self.dtype = torch.float32
        self.unet.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)
        self.bridge_noise_sigma = self.cfg.bridge_noise_sigma

        self.test_data_loader = get_i2i_3D_dataloader(self.test_csv_path, root_dir=self.cfg.data_dir, stage="test", batch_size=1)

        self.unet.eval()
        if self.use_ema:
            self.ema.apply_shadow(self.unet)



    def load_unet(self):
        # load state dict from ckpt 
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
        if self.use_ema:
            self.ema = EMA(self.cfg.EMA.ema_decay)
            self.update_ema_interval = self.cfg.EMA.update_ema_interval
            self.start_ema_step = self.cfg.EMA.start_ema_step
            # self.ema.register(self.unet)
            self.ema.shadow = ckpt['ema_state']
            self.ema.reset_device(self.unet)
        
        return unet_model


    def test(self):
        
        metrics = {}

        for sample in self.test_data_loader:
            src_img = sample['A'].to(self.device, dtype=self.dtype)
            if self.vae is not None:
                z = self.vae.encode_stage_2_inputs(src_img)
            else:
                z = src_img

            sample_id = sample['ID']
            if self.save_img_output:
                os.makedirs(os.path.join(self.save_img_dir, sample_id), exist_ok=True)

                for i in range(self.num_outputs_per_sample):
                    decoded_sample = self.sample(z)
                    # convert to numpy and save as .npy 
                    decoded_sample = decoded_sample.cpu().numpy()
                    np.save(os.path.join(self.save_img_dir, sample_id, f"output_{i}.npy"), decoded_sample)

                    if i == 0 and self.compute_metrics:
                        target_img = sample['B'].to(self.device, dtype=self.dtype)
                        fid = self.evaluate_fid(decoded_sample, target_img)
                        ssim = self.evaluate_ssim(decoded_sample, target_img)
                        lpips = self.evaluate_lpips(decoded_sample, target_img)
                        psnr = self.evaluate_psnr(decoded_sample, target_img)
                        metrics[sample_id] = {
                            "fid": fid,
                            "ssim": ssim,
                            "lpips": lpips,
                            "psnr": psnr,
                        }

            
        return metrics

    def evaluate_3D_fid(self, decoded_sample, target_img):
        # TODO 
        pass

    def evaluate_ssim(self, decoded_sample, target_img):
        # TODO 
        pass

    def evaluate_lpips(self, decoded_sample, target_img):
        # TODO 
        pass

    def evaluate_psnr(self, decoded_sample, target_img):
        # TODO 
        pass
        
    def _get_conditioning(self,z: torch.Tensor):
        """
        Get the conditionings
        861637277698-75joocljpflupsabmsiqpu5hgiep974i.apps.googleusercontent.com](http://861637277698-75joocljpflupsabmsiqpu5hgiep974i.apps.googleusercontent.com
        """
        
        batch_size = z.shape[0]
        conditioning = torch.zeros(batch_size, 1, 4).to(self.device, dtype=self.dtype)
        return conditioning

    def _get_sigmas(
            self, scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cpu"
        ):
            sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
            schedule_timesteps = scheduler.timesteps.to(device)
            timesteps = timesteps.to(device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

    def sample( self, z: torch.Tensor, num_steps: int = 4,
        verbose: bool = False,
    ):
        self.sampling_noise_scheduler.set_timesteps(
            sigmas=np.linspace(1, 1 / num_steps, num_steps)
        )

        sample = z

        # Get conditioning
        conditioning = self._get_conditioning(z)


        for i, t in tqdm(
            enumerate(self.sampling_noise_scheduler.timesteps), disable=not verbose
        ):
        
            denoiser_input = sample

            # Predict noise level using denoiser using conditionings
            pred = self.unet(
                sample=denoiser_input,
                timesteps=t.to(z.device).repeat(denoiser_input.shape[0]),
                context=conditioning,
            )

            # Make one step on the reverse diffusion process
            sample = self.sampling_noise_scheduler.step(
                pred, t, sample, return_dict=False
            )[0]
            if i < len(self.sampling_noise_scheduler.timesteps) - 1:
                timestep = (
                    self.sampling_noise_scheduler.timesteps[i + 1]
                    .to(z.device)
                    .repeat(sample.shape[0])
                )
                sigmas = self._get_sigmas(
                    self.sampling_noise_scheduler, timestep, n_dim=len(z.shape), device=z.device
                )
                sample = sample + self.bridge_noise_sigma * (
                    sigmas * (1.0 - sigmas)
                ) ** 0.5 * torch.randn_like(sample)
                sample = sample.to(z.dtype)

        if self.vae is not None:
            decoded_sample = self.vae.decode(sample)

        else:
            decoded_sample = sample

        return decoded_sample

if __name__ == "__main__":
    inference = Inference(args)
    inference.test()