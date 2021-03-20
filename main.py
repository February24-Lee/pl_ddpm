import torch

from src.GaussianDiffusion_pl import GaussianDiffusion
from src.model import UNet
from src.callbacks import Sampling, SaveCheckpoint

from src.dataloader.celebA_HQ import celebA_HQ_DataModule

import argparse, yaml

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('-c', '--config', required=True)
    parse.add_argument('-g', '--gpus', type=int)
    
    args = parse.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    #logger
    if 'test_tube_params' in config:
        logger = TestTubeLogger(**config['test_tube_params'])
    else:
        logger = None
    
    #Sample
    x_T = torch.randn(config['sample_num'], 
                    config['model_params']['in_ch'],
                    config['data_module']['data_size'],
                    config['data_module']['data_size'])
    
    #DataLoader
    celebA_HQ = celebA_HQ_DataModule(**config['data_module'])
    
    #Callbacks
    callbacks =[]
    if 'callback_sample' in config:
        callbacks.append(Sampling(**config['callback_sample'],
                                x_T=x_T))
        
    if 'callback_savecheckpoint' in config:
        callbacks.append(SaveCheckpoint(
            **config['callback_savecheckpoint'],
            x_T=x_T))
    
    #backbone
    unet = UNet(**config['model_params'])
    
    # ddpm model
    gaussian_module = GaussianDiffusion(unet,
                                        **config['gaussian_ddpm_params'])

    trainer = Trainer(**config['trainer_params'],
                      logger = logger,
                      gpus=args.gpus,
                      callbacks=callbacks)
    
    trainer.fit(gaussian_module, celebA_HQ)
    
