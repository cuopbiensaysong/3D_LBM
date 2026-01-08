from trainer import Trainer
import omegaconf
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler


# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached")

class LBMTrainer(Trainer):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self.source_key = "A"
        self.target_key = "B"
        self.latent_loss_type = cfg.lbm.latent_loss_type
        self.latent_loss_weight = cfg.lbm.latent_loss_weight
        self.pixel_loss_type = cfg.lbm.pixel_loss_type
        self.pixel_loss_weight = cfg.lbm.pixel_loss_weight
        self.timestep_sampling = cfg.lbm.timestep_sampling
        self.logit_mean = cfg.lbm.logit_mean
        self.logit_std = cfg.lbm.logit_std
        # Ensure list types (OmegaConf ListConfig -> python list) for custom_timesteps
        self.prob = (
            list(cfg.lbm.prob)
            if isinstance(cfg.lbm.prob, omegaconf.ListConfig)
            else cfg.lbm.prob
        )
        self.selected_timesteps = (
            list(cfg.lbm.selected_timesteps)
            if isinstance(cfg.lbm.selected_timesteps, omegaconf.ListConfig)
            else cfg.lbm.selected_timesteps
        )
        self.bridge_noise_sigma = cfg.lbm.bridge_noise_sigma

        if self.timestep_sampling == "log_normal":
            assert isinstance(self.logit_mean, float) and isinstance(
                self.logit_std, float
            ), "logit_mean and logit_std should be float for log_normal timestep sampling"

        if self.timestep_sampling == "custom_timesteps":
            print(self.selected_timesteps, self.prob)
            assert isinstance(self.selected_timesteps, list) and isinstance(
                self.prob, list
            ), "timesteps and prob should be list for custom_timesteps timestep sampling"
            assert len(self.selected_timesteps) == len(
                self.prob
            ), "timesteps and prob should be of same length for custom_timesteps timestep sampling"
            assert (
                sum(self.prob) == 1
            ), "prob should sum to 1 for custom_timesteps timestep sampling"

        self.sampling_noise_scheduler  = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
        )
        self.training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="scheduler",
        )
        
    

    
    def loss_fn(self, batch, stage='train'):
        print(f"[DEBUG loss_fn] Moving batch to device...", flush=True)
        batch = {k: v.to(self.device, dtype=self.dtype) for k, v in batch.items()}
        print(f"[DEBUG loss_fn] Batch on device.", flush=True)
        if self.vae is not None:
            vae_inputs = batch[self.target_key]
            print(f"[DEBUG loss_fn] VAE input shape: {vae_inputs.shape}, dtype: {vae_inputs.dtype}, device: {vae_inputs.device}", flush=True)
            print(f"[DEBUG loss_fn] Calling VAE encode...", flush=True)
            with torch.no_grad():  # VAE encoding doesn't need gradients
                z = self.vae.encode_stage_2_inputs(vae_inputs)
            print(f"[DEBUG loss_fn] Target encoded, z shape: {z.shape}", flush=True)
            # downsampling_factor = self.vae.downsampling_factor
        else:
            z = batch[self.target_key]
            # downsampling_factor = 1

        source_image = batch[self.source_key]

        if self.vae is not None:
            print(f"[DEBUG loss_fn] Encoding source...", flush=True)
            with torch.no_grad():
                z_source = self.vae.encode_stage_2_inputs(source_image)
            print(f"[DEBUG loss_fn] Source encoded, z_source shape: {z_source.shape}", flush=True)
        else:
            z_source = source_image
        
        conditioning = self._get_conditioning(batch) 

        timestep = self._timestep_sampling(n_samples=z.shape[0], device=z.device) # checked
        sigmas = None # ? check sigmas

        # Create interpolant
        sigmas = self._get_sigmas(
            self.training_noise_scheduler, timestep, n_dim=len(z.shape), dtype=z.dtype, device=z.device
        ) # TODO check sigmas
        noisy_sample = (
            sigmas * z_source
            + (1.0 - sigmas) * z
            + self.bridge_noise_sigma
            * (sigmas * (1.0 - sigmas)) ** 0.5
            * torch.randn_like(z)
        )

        for i, t in enumerate(timestep):
            if t.item() == self.training_noise_scheduler.timesteps[0]:
                noisy_sample[i] = z_source[i]

        prediction = self.unet(
            noisy_sample,
            timesteps=timestep,
            context=conditioning,
        )

        target = z_source - z
        # denoised_sample = noisy_sample - prediction * sigmas # chuyen xuong duoi 
        target_pixels = batch[self.target_key]

        # Compute loss
        if self.latent_loss_weight > 0:
            # loss = self.masked_latent_loss(prediction, target.detach(), valid_mask_for_latent)
            loss = self.latent_loss(prediction, target.detach())
            latent_recon_loss = loss.mean()

        else:
            loss = torch.zeros(z.shape[0], device=z.device)
            latent_recon_loss = torch.zeros_like(loss)

        if self.pixel_loss_weight > 0:
            denoised_sample = self._predicted_x_0(
                model_output=prediction,
                sample=noisy_sample,
                sigmas=sigmas,
            )
            # pixel_loss = self.masked_pixel_loss(
            #     denoised_sample, target_pixels.detach(), valid_mask
            # )
            pixel_loss = self.pixel_loss(
                denoised_sample, target_pixels.detach()
            )
            loss += self.pixel_loss_weight * pixel_loss

        else:
            pixel_loss = torch.zeros_like(latent_recon_loss)
            denoised_sample = noisy_sample - prediction * sigmas

        # log loss to wandb
        if stage == 'train':
            self.wandb_run.log({
                "step/train_loss": loss.mean().item(),
                "step/epoch": self.global_epoch,
                "step/step": self.global_step,
                "step/latent_recon_loss": latent_recon_loss.mean().item(),
                "step/pixel_recon_loss": pixel_loss.mean().item(),
            })

        return loss.mean()

    def _predicted_x_0(
        self,
        model_output,
        sample,
        sigmas=None,
    ):
        """
        Predict x_0, the orinal denoised sample, using the model output and the timesteps depending on the prediction type.
        """
        pred_x_0 = sample - model_output * sigmas
        return pred_x_0

    def latent_loss(self, prediction, model_input):
        if self.latent_loss_type == "l2":
            return torch.mean(
                (
                    (prediction - model_input)
                    ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )
        elif self.latent_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    prediction - model_input
                ).reshape(model_input.shape[0], -1),
                1,
            )
        else:
            raise NotImplementedError(
                f"Loss type {self.latent_loss_type} not implemented"
            )

    def pixel_loss(self, prediction, model_input):
        
        decoded_prediction = self.vae.decode_stage_2_outputs(prediction)# .clamp(-1, 1) # TODO check decoded_prediction

        if self.pixel_loss_type == "l2":
            return torch.mean(
                (
                    (decoded_prediction - model_input) ** 2
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "l1":
            return torch.mean(
                torch.abs(
                    decoded_prediction - model_input
                ).reshape(model_input.shape[0], -1),
                1,
            )

        elif self.pixel_loss_type == "lpips":
            return self.lpips_loss(
                decoded_prediction, model_input
            ).mean()



    def _get_conditioning(self,batch: Dict[str, Any]):
        """
        Get the conditionings
        """
        
        batch_size = batch[self.target_key].shape[0]
        conditioning = torch.zeros(batch_size, 1, 4).to(self.device, dtype=self.dtype)
        return conditioning

    def _timestep_sampling(self, n_samples=1, device="cpu"):
        if self.timestep_sampling == "uniform":
            idx = torch.randint(
                0,
                self.training_noise_scheduler.config.num_train_timesteps,
                (n_samples,),
                device="cpu",
            )
            return self.training_noise_scheduler.timesteps[idx].to(device=device)

        elif self.timestep_sampling == "log_normal":
            u = torch.normal(
                mean=self.logit_mean,
                std=self.logit_std,
                size=(n_samples,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (
                u * self.training_noise_scheduler.config.num_train_timesteps
            ).long()
            return self.training_noise_scheduler.timesteps[indices].to(device=device)

        elif self.timestep_sampling == "custom_timesteps":
            idx = np.random.choice(len(self.selected_timesteps), n_samples, p=self.prob)

            return torch.tensor(
                self.selected_timesteps, device=device, dtype=torch.long
            )[idx]

    
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




if __name__ == "__main__":
    cfg = omegaconf.OmegaConf.load("configs/lbm_cf.yaml")
    trainer = LBMTrainer(cfg)
    trainer.train()