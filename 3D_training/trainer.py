import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

from tqdm.autonotebook import tqdm

# data loader
from data_3D_loader import get_i2i_3D_dataloader
# backbone
from backbone import get_autoencoder_model_from_ckpt, get_unet_model_from_ckpt
# utils
from utils import setup_wandb, save_config, get_optimizer
import omegaconf
from ema import EMA
import torch

class Trainer():

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        # wandb to log 
        self.wandb_run = setup_wandb(cfg.project_name, cfg.run_name, cfg)
        self.save_dir = os.path.join(cfg.save_dir, cfg.run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        save_config(cfg, self.save_dir)

        # backbone
        self.vae = get_autoencoder_model_from_ckpt(cfg.vae_path)
        self.unet = get_unet_model_from_ckpt(cfg.pretrained_unet_path)

        self.global_epoch = 0  # global epoch
        self.global_step = 0

        self.optimizer, self.scheduler = self.initialize_optimizer_scheduler()
        self.use_ema = cfg.EMA.use_ema
        if self.use_ema:
            self.ema = EMA(cfg.EMA.ema_decay)
            self.update_ema_interval = cfg.EMA.update_ema_interval
            self.start_ema_step = cfg.EMA.start_ema_step
            self.ema.register(self.unet)
        
        self.resume_state()

        if self.cfg.use_bfloat16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.unet.to(self.device, dtype=self.dtype)
        self.vae.to(self.device, dtype=self.dtype)

        # data loader
        self.train_loader = get_i2i_3D_dataloader(cfg.train_csv_path, root_dir=cfg.data_dir, stage="train", batch_size=cfg.batch_size, num_workers=cfg.num_workers, cache_rate=cfg.cache_rate)
        self.val_loader = get_i2i_3D_dataloader(cfg.val_csv_path, root_dir=cfg.data_dir, stage="val", batch_size=cfg.batch_size, num_workers=cfg.num_workers, cache_rate=cfg.cache_rate)



    def initialize_optimizer_scheduler(self):
        optimizer = get_optimizer(self.cfg.optimizer, self.unet.parameters())

        lr_sched_cfg = self.cfg.lr_scheduler
        if isinstance(lr_sched_cfg, (omegaconf.DictConfig, omegaconf.ListConfig)):
            lr_sched_cfg = omegaconf.OmegaConf.to_container(lr_sched_cfg, resolve=True)
        lr_sched_kwargs = {k: v for k, v in lr_sched_cfg.items() if not str(k).startswith("_")}

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer=optimizer,
                                        mode='min',
                                        verbose=True,
                                        threshold_mode='rel',
                                        **lr_sched_kwargs
                    )
        return optimizer, scheduler


    # EMA part
    def step_ema(self):
        with_decay = False if self.global_step < self.start_ema_step else True
        
        self.ema.update(self.unet, with_decay=with_decay)

    def apply_ema(self):
        if self.use_ema:
            self.ema.apply_shadow(self.unet)

    def restore_ema(self):
        if self.use_ema:
            self.ema.restore(self.unet)

    def train(self):
        print(f"start training {self.cfg.run_name} on {self.cfg.train_csv_path}, {len(self.train_loader)} iters per epoch")
        accumulate_grad_batches = self.cfg.accumulate_grad_batches
        best_val_loss = float('inf')
        for epoch in range(self.global_epoch, self.cfg.n_epochs):
            print(f"[DEBUG] Starting epoch {epoch}, creating progress bar...")
            pbar = tqdm(self.train_loader, total=len(self.train_loader), smoothing=0.01)
            self.global_epoch = epoch
            epoch_loss = 0
            print("[DEBUG] Starting iteration over train_loader...")
            for batch_idx, train_batch in enumerate(pbar):
                if batch_idx == 0:
                    print(f"[DEBUG] First batch received! Keys: {train_batch.keys() if hasattr(train_batch, 'keys') else type(train_batch)}")
                self.global_step += 1
                self.unet.train()
                loss = self.loss_fn(
                                    batch=train_batch,
                                    stage='train')

                loss.backward()
                if self.global_step % accumulate_grad_batches == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step(loss)
                if self.use_ema and self.global_step % (self.update_ema_interval*accumulate_grad_batches) == 0:
                    self.step_ema()
                pbar.set_description(
                    (
                        f'Epoch: [{epoch + 1} / {self.cfg.n_epochs}] '
                        f'iter: {self.global_step} loss: {loss.item():.4f}'
                    )
                )
                epoch_loss += loss.item()
                
            
            
            epoch_loss /= len(self.train_loader)
            self.wandb_run.log({
                "epoch/train_ep_loss": epoch_loss,
                "epoch/epoch": epoch,
                "epoch/step": self.global_step,
                "epoch/lr" : self.scheduler.get_last_lr()[0]

            })

            # validation
            if (epoch + 1) % self.cfg.validation_interval == 0 or (
                    epoch + 1) == self.cfg.n_epochs:
                with torch.no_grad():
                    print("validating epoch...")
                    average_loss = self.validation_epoch(epoch)
                    torch.cuda.empty_cache()
                    print("validating epoch success")
                    self.wandb_run.log({
                        "epoch/val_loss": float(average_loss),
                    })
            # save best checkpoint based on validation loss
            if average_loss < best_val_loss:
                best_val_loss = average_loss
                self.save_state(epoch, "best_checkpoint.pth")

            #  Save last checkpoint
            self.save_state(epoch, "last_checkpoint.pth")


    def validation_epoch(self, epoch: int):
        self.apply_ema()
        self.unet.eval()

        pbar = tqdm(self.val_loader, total=len(self.val_loader), smoothing=0.01)
        step = 0
        loss_sum = 0.0
        for val_batch in pbar:
            loss = self.loss_fn(batch=val_batch,stage='val')
            # wandb cannot serialize MetaTensor; accumulate as plain python float
            loss_val = loss.item() if hasattr(loss, "item") else float(loss)
            loss_sum += loss_val
            step += 1
        average_loss = loss_sum / step
        self.restore_ema()
        return average_loss
        
    def loss_fn(self, batch, stage='train'):
        """
        loss function
        :param net: nn.Module
        :param batch: batch
        :param epoch: global epoch
        :param step: global step
        :param opt_idx: optimizer index, default is 0; set it to 1 for GAN discriminator
        :param stage: train, val, test
        :param write: write loss information to SummaryWriter
        :return: a scalar of loss
        """
        pass


    def save_state(self, epoch:int, filename: str = "state.pth"):
        checkpoint = {
            "epoch": epoch,
            "unet_state": self.unet.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if self.use_ema:
            checkpoint["ema_state"] = self.ema.shadow

        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        checkpoint_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)

    def resume_state(self):
        model_states = None 
        if self.cfg.__contains__("resume_state_path") and self.cfg.resume_state_path is not None:

            model_states = torch.load(self.cfg.resume_state_path)
            self.unet.load_state_dict(model_states["unet_state"], strict=True)
            self.optimizer.load_state_dict(model_states["optimizer_state"], strict=True)
            if self.use_ema:
                self.ema.shadow = model_states['ema']
                self.ema.reset_device(self.unet)

            self.global_epoch = model_states["epoch"]
            self.global_step = model_states["global_step"]
            if self.scheduler is not None:
                self.scheduler.load_state_dict(model_states["scheduler_state"], strict=True)

        return model_states
