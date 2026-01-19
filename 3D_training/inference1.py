import argparse
import os
import omegaconf
import torch

from generative.networks.nets import DiffusionModelUNet
from backbone import get_autoencoder_model_from_ckpt
from ema import EMA
from tqdm import tqdm
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
import pandas as pd

from data_3D_loader import get_i2i_3D_dataloader

# Metric Imports
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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
        self.compute_metrics = args.compute_metrics

        if self.save_img_output:
            self.save_img_dir = os.path.join(self.output_dir, "images")
            os.makedirs(self.save_img_dir, exist_ok=True)
        else:
            self.save_img_dir = None
        
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

        self.test_data_loader = get_i2i_3D_dataloader(self.test_csv_path, root_dir=self.cfg.data_dir, stage="test", batch_size=1)

        self.unet.eval()
        if self.use_ema:
            self.ema.apply_shadow(self.unet)

        # --- Initialize Metrics ---
        if self.compute_metrics:
            print("Initializing Metrics...")
            # FID: Calculates distribution distance. Requires 3 channels, uint8 [0, 255].
            # We will flatten 3D volumes to 2D slices for this.
            self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(self.device)
            
            # LPIPS: Perceptual loss. VGG is standard.
            self.lpips_metric = lpips.LPIPS(net='vgg').to(self.device)
            
            # SSIM: Structural Similarity.
            # self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
            
            # PSNR: Peak Signal to Noise Ratio
            # self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

            # Storage for scalar metrics to average later
            self.scores = {
                "ssim": [],
                "lpips": [],
                "psnr": []
            }


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

    def _prepare_for_2d_metrics(self, volume):
        """
        Convert 5D volume (B, C, D, H, W) to 4D batch of RGB slices (B*D, 3, H, W).
        This allows us to use standard pre-trained 2D metrics (FID-Inception, LPIPS-VGG) on 3D medical data.
        """
        b, c, d, h, w = volume.shape
        # Permute to (B, D, C, H, W) and reshape to (B*D, C, H, W)
        slices = volume.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
        
        # If single channel, repeat to 3 channels for ImageNet-trained networks
        if c == 1:
            slices = slices.repeat(1, 3, 1, 1)
        
        return slices
    
    def _normalize_min_max(self, x):
        """Normalize data to [0, 1] for metrics."""
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    def get_min_max(self, x, y):
        return min(x.min(), y.min()), max(x.max(), y.max())

    def update_metrics(self, decoded_sample, target_img):
        """
        Accumulates statistics for metrics.
        decoded_sample: (B, 1, D, H, W)
        target_img: (B, 1, D, H, W)
        """
        # Ensure data is normalized [0, 1] for metrics consistency
        decoded_norm = self._normalize_min_max(decoded_sample)
        target_norm = self._normalize_min_max(target_img)

        # 1. SSIM (Can handle 5D input with torchmetrics if data_range is set)
        # ssim_metric = structural_similarity(decoded_norm, target_norm)
        # ssim_val = ssim_metric(decoded_norm, target_norm)
        decoded_sample = decoded_sample.cpu().detach().numpy()
        target_img = target_img.cpu().detach().numpy()
        MIN, MAX = self.get_min_max(decoded_sample, target_img)
        
        ssim_val = structural_similarity(decoded_sample[0][0], target_img[0][0], data_range=MAX - MIN)
        self.scores["ssim"].append(ssim_val)

        # 2. PSNR
        psnr_val = peak_signal_noise_ratio(decoded_sample[0][0], target_img[0][0], data_range=MAX - MIN)
        self.scores["psnr"].append(psnr_val)

        # Prepare for 2D-based metrics (LPIPS, FID)
        # Flatten depth into batch dimension: (B*D, 3, H, W)
        decoded_2d = self._prepare_for_2d_metrics(decoded_norm)
        target_2d = self._prepare_for_2d_metrics(target_norm)

        # 3. LPIPS (Calculated per slice pair, then averaged for the volume)
        # LPIPS expects input in range [-1, 1] usually, but our logic passes [0, 1]. 
        # lpips library handles normalization if configured, but VGG expects specific normalization.
        # Standard approach: inputs in [-1, 1].
        decoded_lpips_in = (decoded_2d * 2) - 1
        target_lpips_in = (target_2d * 2) - 1
        
        with torch.no_grad():
            lpips_val = self.lpips_metric(decoded_lpips_in, target_lpips_in)
        self.scores["lpips"].append(lpips_val.mean().item())

        # 4. FID (Update state, don't compute yet)
        # FID expects [0, 1] float (if normalize=True in init) or [0, 255] uint8.
        # We initialized with normalize=True.
        # Note: We treat every slice as an independent sample from the distribution.
        self.fid_metric.update(target_2d, real=True)
        self.fid_metric.update(decoded_2d, real=False)

    def test(self):
        print(f"Starting inference on {len(self.test_data_loader)} samples...")
        
        for i, sample in enumerate(tqdm(self.test_data_loader)):
            
            if self.save_img_output:
                save_path = os.path.join(self.save_img_dir, sample_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                else:
                    print(f"Save path {save_path} already exists")
                    continue
                
            src_img = sample['A'].to(self.device, dtype=self.dtype)
            
            # Encode
            if self.vae is not None:
                z = self.vae.encode_stage_2_inputs(src_img)
            else:
                z = src_img

            sample_id = sample['ID'][0] if isinstance(sample['ID'], list) else sample['ID']
            
            # Generate
            # We assume batch_size=1 here based on loader
            for j in range(self.num_outputs_per_sample):
                decoded_sample = self.sample(z, num_steps=self.num_inference_steps)
            # Save Output
                if self.save_img_output:
                    np_sample = decoded_sample.cpu().detach().numpy()
                    np.save(os.path.join(save_path, f"output_{j}.npy"), np_sample)

            # Compute Metrics (Accumulate)
                if self.compute_metrics and j == 0:
                    target_img = sample['B'].to(self.device, dtype=self.dtype)
                    self.update_metrics(decoded_sample, target_img)

        # Finalize Metrics Calculation
        if self.compute_metrics:
            print("Computing final metrics...")
            final_results = {}
            
            # Average per-sample metrics
            final_results["SSIM"] = np.mean(self.scores["ssim"])
            final_results["PSNR"] = np.mean(self.scores["psnr"])
            final_results["LPIPS"] = np.mean(self.scores["lpips"])
            
            # Compute distributional metric exactly over the whole dataset
            # This runs the Fréchet distance calc on the accumulated covariance matrices
            try:
                final_results["FID"] = self.fid_metric.compute().item()
            except Exception as e:
                print(f"Error computing FID (requires >1 sample usually): {e}")
                final_results["FID"] = -1.0

            print("\n=== Final Evaluation Results ===")
            for k, v in final_results.items():
                print(f"{k}: {v:.4f}")
            
            from compute_diversity import calc_diversity
            diversity = calc_diversity(self.save_img_dir)

            final_results["diversity"] = diversity
            # Save to CSV
            df = pd.DataFrame([final_results])
            df.to_csv(os.path.join(self.output_dir, "metrics.csv"), index=False)
            
            return final_results
            
        return None

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
    inference.test()
    
