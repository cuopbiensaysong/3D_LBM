import os
os.environ['WANDB_API_KEY'] = '9ab49432fdba1dc80b8e9b71d7faca7e8b324e3e'
import wandb
import yaml
import torch
import torch.nn as nn
import omegaconf


def setup_wandb(project_name: str, run_name: str, config: dict):
    """
    Initialize wandb if available. Returns wandb.run or None when wandb is missing.
    """
    def _to_plain_config(config):
        """
        Convert OmegaConf configs to plain Python containers so wandb can sanitize
        them without failing on DictConfig/ListConfig types.
        """
        if isinstance(config, (omegaconf.DictConfig, omegaconf.ListConfig)):
            return omegaconf.OmegaConf.to_container(config, resolve=True)
        return config

    plain_config = _to_plain_config(config)
    return wandb.init(project=project_name, name=run_name, config=plain_config, reinit=True)

def save_config(config: dict, run_dir: str, filename: str = "config.yaml"):
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, filename)
    if isinstance(config, (omegaconf.DictConfig, omegaconf.ListConfig)):
        # Preserve OmegaConf types and interpolation when writing the config
        omegaconf.OmegaConf.save(config, config_path)
    else:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
    return config_path

def get_optimizer(optim_config, parameters):
    if optim_config.optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay,
                                betas=(optim_config.beta1, 0.999))
    elif optim_config.optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=optim_config.lr, weight_decay=optim_config.weight_decay)
    elif optim_config.optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=optim_config.lr, momentum=0.9)
    else:
        return NotImplementedError('Optimizer {} not understood.'.format(optim_config.optimizer))


if __name__ == "__main__":
    import numpy as np
    num_steps = 20
    print (np.linspace(1, 1 / num_steps, num_steps))