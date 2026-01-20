import os
# from PIL import Image
import numpy as np
import torch
from tqdm.autonotebook import tqdm


def calc_diversity(data_dir, num_samples=5):
    dir_list = os.listdir(data_dir)
    dir_list.sort()

    total = len(dir_list)
    std = 0
    for dir_name in tqdm(dir_list):
        imgs = []
        for j in range(num_samples):
            # img = Image.open(os.path.join(os.path.join(data_dir, str(i), f'output_{str(j)}.png')))
            np_img = np.load(os.path.join(os.path.join(data_dir, dir_name, f'output_{str(j)}.npy')))
            # img = transform(img)
            imgs.append(torch.from_numpy(np_img))

        img_mean = torch.zeros_like(imgs[0])
        for j in range(num_samples):
            img_mean = img_mean + imgs[j]
        img_mean = img_mean / num_samples

        img_var = torch.zeros_like(imgs[0])
        for j in range(num_samples):
            img_var = img_var + (imgs[j] - img_mean)**2
        img_var = img_var / num_samples
        img_std = torch.sqrt(img_var)
        std = std + torch.mean(img_std)
    std = std / total
    print(data_dir)
    print(f'diversity: {std}')

    return std