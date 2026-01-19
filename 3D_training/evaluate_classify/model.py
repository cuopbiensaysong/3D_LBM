from monai.networks.nets import DenseNet121
import torch 


def get_densenet121(
    ckpt_path='results/from_scratch_DenseNet121_train+val/best_model.pth', 
    device='cuda',
    ): 
    model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
    model.eval()
    model.to(device)
    return model

