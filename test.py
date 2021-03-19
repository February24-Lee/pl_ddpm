import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger

from src.GaussianDiffusion_pl import GaussianDiffusion
from src.model import UNet
from src.callbacks import Sampling, SaveCheckpoint, evaluate_FID_IS

x_T = torch.rand([10, 3, 32, 32])

module_params = {
    "beta_1" : 0.0001,
    "beta_T" : 0.02,
    "T" : 1000,
    "img_size" : 32,
    "mean_type" : 'epsilon',
    "var_type" : 'fixedlarge',
    "optim_lr" : 0.001,
    "warmup" :  10,
    "grad_clip"  : 1,
    "ema_decay" : 0.999}

model_params = {
    "in_ch" : 3,
    "base_ch" : 32,
    "ch_mult" : [1, 1, 2, 3],
    "attn_list" : [1],
    "n_res_block" : 2,
    "dropout_rate" : 0.5,
    "T" : 1000,
    "tdim" : 32,
    "n_groupnorm":  16}

callback_sampling_params={
    "sample_step" : 1,
    "sample_dir" : 'logs/',
    "x_T" : x_T
}
callback_save_params={
    "save_step" : 1,
    "save_dir" : 'logs/',
    "x_T" : x_T
}
test_tube_params={
    "save_dir" : 'logs/',
    "name" : 'test' 
}

class testLoader(Dataset):
    def __init__(self):
        super().__init__()
        self.x = torch.rand([10, 3, 32, 32])
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]
    
logger = TestTubeLogger(**test_tube_params)    

callback_sample = Sampling(**callback_sampling_params)
callback_save = SaveCheckpoint(**callback_save_params)

test_ds = testLoader()
test_loader = DataLoader(test_ds, batch_size=2)
valid_loader = DataLoader(test_ds, batch_size=2)

model = UNet(**model_params)

print("==== model ====")
print(model)
print("================")

test_module = GaussianDiffusion(model,
                                **module_params)


trainer = Trainer(max_epochs=5,
                callbacks=[callback_sample, callback_save],
                logger=logger)
trainer.fit(test_module, test_loader, valid_loader)
