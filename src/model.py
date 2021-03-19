import torch
import pytorch_lightning as pl
from torch import nn

from typing import List
import sys

from .layers import *
from .utils import compose

'''
Implement the DDPM(Denoising Diffusion Probability Model)
Using by pytorch_lightning
'''

class UNet(nn.Module):
    '''
    
    '''
    def __init__(self,
                in_ch:int = 3,
                base_ch: int = 128,
                ch_mult: List[int] = None,
                attn_list: List[int] = None,
                n_res_block: int = 2,
                dropout_rate: float = 0.5,
                T: int = None,
                tdim: int = None,
                n_groupnorm: int = 32):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn_list]), 'attn index out of bound'
        self.timembedding = TimeEmbedding(T, base_ch, tdim)
        
        self.head = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.downblocks = nn.ModuleList()
        chs = [base_ch]
        now_ch = base_ch
        
        for i, mult in enumerate(ch_mult):
            out_ch = base_ch * mult
            for _ in range(n_res_block):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout_rate, is_attn=(i in attn_list)))
                now_ch = out_ch
                chs.append(now_ch)
            
            if i != len(ch_mult) -1 :
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)
                
        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout_rate, is_attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout_rate, is_attn=False),
        ])
        
        self.upblocks = nn.ModuleList()
        for idx, mult in reversed(list(enumerate(ch_mult))):
            out_ch = base_ch * mult
            for _ in range(n_res_block +1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout_rate, is_attn=(idx in attn_list)))
                now_ch = out_ch
                
            if idx != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0
        
        self.tail = nn.Sequential(
            nn.GroupNorm(n_groupnorm, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, 1, 1)
        )
        self.initialize()
        
    def initialize(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        nn.init.zeros_(self.tail[-1].bias)

    def forward(self, x : torch.Tensor, t : torch.Tensor) -> torch.Tensor:
        temb = self.timembedding(t)
        
        h = self.head(x)
        hs : List[torch.Tensor] = [h]
        # --- downsampling
        for layer in self.downblocks:
            if isinstance(layer, DownSample):
                h = layer(h) 
            elif isinstance(layer, ResBlock):
                h = layer(h, temb)  
            else :
                raise  Exception('Not Implemented Layer')
            hs.append(h)
            
        # --- middle
        for layer in self.middleblocks:
            h = layer(h, temb)
            
        # --- Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
                h = layer(h, temb)
            elif isinstance(layer, UpSample):
                h = layer(h)
            else:
                raise Exception('Not Implemented Layer')
        h = self.tail(h)
        
        assert  len(hs) == 0
        return h
        
if __name__ == '__main__':
    model = UNet(in_ch=3, base_ch=128, ch_mult=[1, 1, 2, 2],
                attn_list=[1], n_res_block=2,
                dropout_rate=0.1,
                T = 1000,
                tdim=128*4,
                n_groupnorm=32)
    print(model)
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(1000, (2,))
    y = model(x, t)
    print(y.shape)