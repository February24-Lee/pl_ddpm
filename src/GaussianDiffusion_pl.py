import torch, pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

import copy
from typing import List, Dict
import numpy as np

from .utils import extract, ema

MEAN_TYPE = ['xprev', 'xstart', 'epsilon']
VAR_TYPE = ['fixedlarge', 'fixedsmall']

class GaussianDiffusion(pl.LightningModule):
    def __init__(self,
                model : nn.Module = None,
                beta_1 : float = None,
                beta_T : float = None,
                T : int = 1000,
                img_size : int = 32,
                mean_type : str = 'epsilon',
                var_type : str = 'fixedlarge',
                optim_lr : float = 0.001,
                grad_clip : float = 1,
                ema_decay : float = 0.9,
                warmup : int = 100):
        assert (mean_type in MEAN_TYPE) & (var_type in VAR_TYPE)
        super().__init__()
        
        self.save_hyperparameters()
        self.model = model
        self.ema_model = copy.deepcopy(model)
        
        self.T = T
        self.img_size = img_size
        
        self.optim_lr = optim_lr
        self.warmup = warmup
        self.grad_clip = grad_clip
        self.ema_decay = ema_decay
        self.var_type = var_type
        self.mean_type = mean_type
        
        self.register_buffer('betas',                       torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        
        self.register_buffer( 'sqrt_alphas_bar',            torch.sqrt(alphas_bar))
        self.register_buffer( 'sqrt_one_minus_alphas_bar',  torch.sqrt(1. - alphas_bar))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer( 'sqrt_recip_alphas_bar',      torch.sqrt(1. / alphas_bar))
        self.register_buffer( 'sqrt_recipm1_alphas_bar',    torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer( 'posterior_var',              self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer( 'posterior_log_var_clipped',  torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer( 'posterior_mean_coef1',       torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer( 'posterior_mean_coef2',       torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar)) 
        
    def setup(self, stage=None):
        self.logger.log_hyperparams(self.hparams)
        
    def q_mean_variance(self, x_0:torch.Tensor, x_t:torch.Tensor, t:torch.Tensor) -> List[torch.Tensor] :
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = ( extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
                            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t )
        posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped
    
    def predict_xstart_from_eps(self, x_t : torch.Tensor, t : torch.Tensor, eps : torch.Tensor) -> torch.Tensor:
        """
        originally, x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * eps
        """
        assert x_t.shape == eps.shape
        return ( extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps )
        
    def predict_xstart_from_xprev(self, x_t : torch.Tensor, t : torch.Tensor, xprev : torch.Tensor) -> torch.Tensor:
        """
        x_0 = (xprev - coef2*x_t) / coef1
        """
        assert x_t.shape == xprev.shape
        return (   extract( 1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
                    extract( self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t )
        
    def p_mean_variance(self, x_t : torch.Tensor, t : torch.Tensor, model_type :str = 'model') -> torch.Tensor:
        # below: only log_variance is used in the KL computations
        
        # for fixedlarge, we set the initial (log-)variance like so to
        # get a better decoder log likelihood
        assert model_type in ['model', 'ema_model']
        if model_type is 'model':
            model = self.model
        elif model_type is 'ema_model' :
            model = self.ema_model
        
        model_log_var = {
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped, }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = model(x_t, t)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = model(x_t, t)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = model(x_t, t)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)

        return model_mean, model_log_var
        
    def forward(self, x_T : torch.Tensor, model_type:str = 'model') -> torch.Tensor:
        assert model_type in ['model', 'ema_model']
        x_t = x_T
        x_t = x_t.to(self.device)
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, log_var = self.p_mean_variance(x_t=x_t, t=t, model_type=model_type)
            if time_step > 0:
                noise = torch.rand_like(x_t)
            else :
                noise = 0
            x_t = mean + torch.exp(0.5 * log_var) * noise
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
                    
    def training_step(self, batch : torch.Tensor, batch_idx : int) -> torch.Tensor:
        loss = self.share_step(batch, batch_idx) 
        #self.logger.experiment.log(loss, global_step=self.global_step)
        self.logger.log_metrics({'loss' : loss['loss'].item()}, step=self.global_step)
        return loss
    
    def validation_step(self, batch : torch.Tensor, batch_idx : int ) -> torch.Tensor:
        loss = self.share_step(batch, batch_idx)
        return loss
    
    def validation_epoch_end(self, outputs : List[torch.Tensor]) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.logger.log_metrics({'avg_val_loss' : avg_loss.item()}, step=self.current_epoch)

        
    def share_step(self, batch : torch.Tensor, batch_dix : int ) -> Dict[str, torch.Tensor]:
        t = torch.randint(self.T, size=(batch.shape[0], ), device=batch.device)
        noise = torch.randn_like(batch)
        x_t = ( extract(self.sqrt_alphas_bar, t, batch.shape) * batch +
               extract(self.sqrt_one_minus_alphas_bar, t, batch.shape) * noise)
        assert batch.shape == x_t.shape
        loss = F.mse_loss(self.model(x_t, t), noise, reduction='none').mean()
        return {'loss' : loss, 't' : t}
        
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr = self.optim_lr)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda= lambda x : min(x, self.warmup)/self.warmup)
        return [optim], [sched]
    
    def on_after_backward(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
    def on_before_zero_grad(self, optimizer):
        ema(self.model, self.ema_model, self.ema_decay)
