import yaml
import torch
from tqdm import tqdm  # Recommended for tracking progress
from monai.utils import set_determinism

from data_3D_loader import get_vqgan_3D_dataloader
from backbone import get_autoencoder_model_from_ckpt

# for reproducibility purposes set a seed
set_determinism(42)

def main(
    train_txt_path: str = '/home/huutien/5_folds_split_3D/VQGAN/fold_1_train.txt',
    data_dir: str = '/home/huutien/filter_ds',
    device='cuda',
    batch_size=1,
    num_workers=4,
    cache_rate=1.0,
    use_bfloat16: bool = True,
    *args, **kwargs
):

    train_loader = get_vqgan_3D_dataloader(data_dir, train_txt_path, stage="compute", batch_size=batch_size, num_workers=num_workers, cache_rate=cache_rate)
    # val_loader not strictly needed for computing training set stats, but keeping it as per original code
    # val_loader = get_vqgan_3D_dataloader(data_dir, val_txt_path, stage="val", batch_size=batch_size, num_workers=num_workers, cache_rate=cache_rate)

    autoencoder = get_autoencoder_model_from_ckpt('/home/huutien/sources/GenerativeModels/3D_training/results/3D_training_KL_20251229_101810/best_checkpoint.pth')
    autoencoder.to(device)
    # Ensure model is in the correct dtype
    autoencoder.to(dtype=torch.bfloat16 if use_bfloat16 else torch.float32)
    autoencoder.eval()

    # Variables to accumulate statistics
    channel_sum = None
    channel_squared_sum = None
    total_voxels = 0

    print("Starting calculation of latent statistics...")
    
    # Disable gradients for faster processing and lower memory usage
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Computing Stats"):
            images = batch["image"]
            if hasattr(images, "as_tensor"):
                images = images.as_tensor()
            
            images = images.to(device)
            if use_bfloat16:
                images = images.to(dtype=torch.bfloat16)

            # Get latents (z)
            z = autoencoder.encode_stage_2_inputs(images)

            # Important: Cast to float32 for statistical accumulation to avoid overflow/precision issues
            z = z.float()

            # Assuming z shape is [Batch, Channel, Depth, Height, Width]
            # We want stats per Channel, so we aggregate over (Batch, D, H, W)
            # Dimensions to reduce: 0, 2, 3, 4
            reduction_dims = [0, 2, 3, 4]
            
            # Calculate current batch sums
            current_sum = torch.sum(z, dim=reduction_dims)
            current_squared_sum = torch.sum(z ** 2, dim=reduction_dims)
            
            # Calculate number of voxels in this batch (Batch * D * H * W)
            # logic: product of shape at reduction dimensions
            batch_num_voxels = z.shape[0] * z.shape[2] * z.shape[3] * z.shape[4]

            # Accumulate
            if channel_sum is None:
                channel_sum = current_sum
                channel_squared_sum = current_squared_sum
            else:
                channel_sum += current_sum
                channel_squared_sum += current_squared_sum
            
            total_voxels += batch_num_voxels

    # --- Final Calculation ---
    # Mean = Sum / N
    latents_mean = channel_sum / total_voxels

    # Std = Sqrt( E[x^2] - (E[x])^2 )
    # Note: We use clamp(min=0) to prevent tiny negative numbers due to floating point errors
    latents_std = (channel_squared_sum / total_voxels - latents_mean ** 2).clamp(min=0).sqrt()

    print("\n" + "="*30)
    print("STATISTICS COMPUTED")
    print("="*30)
    print(f"Total Voxels Processed: {total_voxels}")
    print(f"Latents Mean (shape {latents_mean.shape}):\n{latents_mean}")
    print(f"Latents Std  (shape {latents_std.shape}):\n{latents_std}")

    # Optional: Save to file for easy loading later
    # torch.save({
    #     'mean': latents_mean,
    #     'std': latents_std
    # }, 'latent_stats.pth')
    # print("Saved stats to 'latent_stats.pth'")

if __name__ == "__main__":
    config_path = './configs/KL_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    main(**config)