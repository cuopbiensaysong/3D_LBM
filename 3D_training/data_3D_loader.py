# import pandas as pd
import os
import torch
import pandas as pd
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    Orientationd,
    Lambdad
)
def get_vqgan_3D_dataloader(data_dir, txt_path, stage='train', batch_size=4, num_workers=4, cache_rate=1.0):
    if stage == 'train':
        shuffle = True
    else:
        shuffle = False

    with open(txt_path, 'r') as f:
        nifti_list = f.readlines()
    nifti_list = [item.strip() for item in nifti_list]
    
    data_dicts = []
    for nifti_path in nifti_list:
        data_dicts.append({
            "image": os.path.join(data_dir, nifti_path.strip()),
        })

    # --- 1. Deterministic Transforms (Cached) ---
    # Moved EnsureTyped here so both Train AND Val data become Tensors.
    pre_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Lambdad(keys=["image"], func=lambda x: x - 0.8),
        EnsureTyped(keys=["image"]), # <--- Moved here
    ])

    # --- 2. Dynamic Augmentations (Train only) ---
    if stage == 'train':
        aug_transforms = Compose([
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.3, spatial_axis=2),
            RandRotate90d(keys=["image"], prob=0.3, max_k=3),
            
            RandAffined(
                keys=["image"],
                mode="bilinear", # <--- FIX: "bilinear" string, or ("bilinear",) tuple
                prob=0.5,
                rotate_range=(0.26, 0.26, 0.26),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border"
            ),

            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        ])
    else:
        aug_transforms = None

    # --- 3. Dataset Setup ---
    print(f"Initializing {stage} dataset with {len(data_dicts)} images...")
    
    # Load and cache base data
    cached_ds = CacheDataset(
        data=data_dicts, 
        transform=pre_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )

    # Wrap with augmentations if training
    if stage == 'train':
        ds = Dataset(data=cached_ds, transform=aug_transforms)
    else:
        ds = cached_ds

    # --- 4. DataLoader ---
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    return loader

def get_i2i_3D_dataloader(csv_path, stage='train', root_dir="./", batch_size=4, num_workers=4, cache_rate=1.0):

    if stage == 'train':
        shuffle = True
    else:
        shuffle = False

    # 1. Parse CSV
    df = pd.read_csv(csv_path)
    data_dicts = []
    for _, row in df.iterrows():
        data_dicts.append({
            "A": os.path.join(root_dir, row["img_path_A"].strip()),
            "B": os.path.join(root_dir, row["img_path_B"].strip()),
            "subject_id": row["subject ID"]
        })

    # 2. STAGE 1: Deterministic Transforms (Cached)
    # These run ONCE and are stored in RAM.
    pre_transforms = Compose([
        LoadImaged(keys=["A", "B"]),
        EnsureChannelFirstd(keys=["A", "B"]),
        # Standardize orientation to RAS (Right, Anterior, Superior) to ensure consistency
        Orientationd(keys=["A", "B"], axcodes="RAS"), 
        # Deterministic intensity scaling
        Lambdad(keys=["A", "B"], func=lambda x: x - 0.8),
        EnsureTyped(keys=["A", "B"]),
    ])

    # 3. STAGE 2: Random Augmentations (On-the-fly)
    # These run on the CPU every time a batch is fetched.
    # Note: We apply geometric transforms to BOTH A and B to keep them aligned.
    if stage == 'train':
        aug_transforms = Compose([
            # --- Spatial Augmentations (Must align A & B) ---
            
            # 1. Random Flip along X, Y, or Z axes
            RandFlipd(keys=["A", "B"], prob=0.3, spatial_axis=0),
            RandFlipd(keys=["A", "B"], prob=0.3, spatial_axis=1),
            RandFlipd(keys=["A", "B"], prob=0.3, spatial_axis=2),
            
            # 2. Random 90 degree rotation (cheap and effective)
            RandRotate90d(keys=["A", "B"], prob=0.3, max_k=3),
            
            # 3. Random Affine (Rotation + Zoom)
            # This is the most powerful 3D augmentation.
            # rotate_range: +/- 15 degrees (in radians approx 0.26)
            # scale_range: zoom in/out by 10%
            RandAffined(
                keys=["A", "B"],
                mode=("bilinear", "bilinear"),
                prob=0.5,
                rotate_range=(0.26, 0.26, 0.26),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border" # Important to avoid black borders
            ),

            # --- Intensity Augmentations (Usually independent) ---
            # Only adding noise to Input (A) helps the model denoise/generalize.
            # If A and B are both noisy modalities, you might want it on both.
            RandGaussianNoised(keys=["A"], prob=0.3, mean=0.0, std=0.1),

            # Final Tensor Conversion
            EnsureTyped(keys=["A", "B"]),
        ])
    else:
        aug_transforms = None

    print(f"Loading {len(data_dicts)} pairs into RAM (Stage 1)...")
    
    # A. The Cached Dataset (Holds the pre-processed tensors in RAM)
    cached_ds = CacheDataset(
        data=data_dicts, 
        transform=pre_transforms, 
        cache_rate=cache_rate, 
        num_workers=num_workers
    )

    # B. The Wrapper Dataset (Applies random augs to the cached tensors)
    # This ensures every epoch sees a NEW variation of the data.
    if stage == 'train':
        ds = Dataset(data=cached_ds, transform=aug_transforms)
    else:
        ds = cached_ds

    # 4. DataLoader
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available()
    )
    
    return loader