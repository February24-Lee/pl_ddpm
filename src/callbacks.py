import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from torch import nn
from torchvision.utils import make_grid, save_image

import os
from tqdm import trange
import json
from pathlib import Path

from .score.both import get_inception_and_fid_score

class Sampling(Callback):
    def __init__(self, sample_step : int = 10, 
                sample_dir : str = '.',
                x_T : torch.Tensor = None):
        super().__init__()
        self.sample_step = sample_step
        self.sample_dir = sample_dir
        self.x_T = x_T
        
    
    def on_train_epoch_end(self, trainer : pl.Trainer, pl_module : pl.LightningModule, outputs):
        if pl_module.current_epoch % self.sample_step == 0:
            # TODO 왜 mopdel을 eval 하고 ema_model에서 샘플링...?
            pl_module.model.eval()
            with torch.no_grad():
                x_0 = pl_module(self.x_T, 'ema_model')
                grid =   (make_grid(x_0) + 1) / 2
                Path(os.path.join( self.sample_dir, 'sample' )).mkdir(parents=True, exist_ok=True)
                path = os.path.join( self.sample_dir, 'sample', '%d.png' % pl_module.current_epoch )
                save_image(grid, path)
            pl_module.model.train()
            
class SaveCheckpoint(Callback):
    def __init__(self, 
                save_step : int = 10, 
                save_dir : str = '.',
                x_T : torch.Tensor = None):
        super().__init__()
        self.save_step = save_step
        self.save_dir = save_dir
        self.x_T = x_T
    
    def on_train_epoch_end(self, trainer : pl.Trainer, pl_module : pl.LightningModule, outputs):
        if pl_module.current_epoch % self.save_step == 0:
            ckpt = {
                'net_model' : pl_module.model.state_dict(),
                'ema_model' : pl_module.ema_model.state_dict(),
                'optim' : pl_module.optimizers().state_dict(),
                'epoch' : pl_module.current_epoch,
                #'x_T' : self.x_T
            }
            Path(os.path.join(self.save_dir)).mkdir(parents=True, exist_ok=True)
            path = os.path.join(self.save_dir, 'ckpt.pt')
            torch.save(ckpt, path)
            

class evaluate_FID_IS(Callback):
    def __init__(self,
                n_images : int = None,
                batch_size : int = None,
                img_size : int = None,
                fid_cache = None,
                fid_use_torch : bool = None,
                fid_verbose : bool = True,
                evaluate_step: int = None,
                log_dir : str = '.'):
        super().__init__()
        self.n_images = n_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.fid_cache = fid_cache
        self.fid_use_torch = fid_use_torch
        self.fid_verbose = fid_verbose
        self.evaluate_step = evaluate_step
        self.log_dir = log_dir
        
    def evaluate(self, pl_module, model, type_model : str = 'model'):
        model.eval()
        with torch.no_grad():
            images = []
            desc = "generating images"
            for i in trange(0, self.n_images, self.batch_size, desc=desc):
                batch_size = min(self.batch_size, self.n_images - i)
                x_T = torch.randn((batch_size, 3, self.img_size, self.img_size))
                batch_images = pl_module(x_T.to(model.device)).cpu()
                images.append((batch_images + 1) / 2)
            images = torch.cat(images, dim=0).numpy()
        model.train()
        (IS, IS_std), FID = get_inception_and_fid_score(
            images, self.fid_cache, num_images=self.n_images,
            use_torch=self.fid_use_torch, verbose=self.fid_verbose)
        return (IS, IS_std), FID, images
    
    def on_train_epoch_end(self, trainer : pl.Trainer, pl_module : pl.LightningModule, outputs):
        if pl_module.current_epoch % self.evaluate_step == 0:
            net_IS, net_FID, _ = self.evaluate(pl_module, pl_module.model, 'model')
            ema_IS, ema_FID, _ = self.evaluate(pl_module, pl_module.ema_model, 'ema_model')
            metrics = {
                'IS': net_IS[0],
                'IS_std': net_IS[1],
                'FID': net_FID,
                'IS_EMA': ema_IS[0],
                'IS_std_EMA': ema_IS[1],
                'FID_EMA': ema_FID}
            with open(os.path.join(self.logdir, 'eval.txt'), 'a') as f:
                metrics['step'] = pl_module.current_epoch
                f.write(json.dumps(metrics) + "\n")
                
                
