import torch
from torch import nn
import torch.nn.functional as F

from math import log

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    '''
    input.
        T (torch.Tensor) : chain-rule의 time
    
    output.
        Time Embedding Vector (tensor.Tensor shape = temdim)
    '''
    def __init__(self, 
                T:torch.Tensor,
                dim_model : int = None,
                dim : int = None):
        super(TimeEmbedding, self).__init__()
        assert dim_model % 2 ==0, 'dimension of model should be even'
        emb = self.make_emb(T, dim_model)
        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(dim_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()
        
    def make_emb(self, size:torch.Tensor, dim_model:int):
        emb = torch.arange(0, dim_model, step=2) / dim_model * log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(size).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [size, dim_model//2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [size, dim_model//2, 2]
        emb = emb.view(size, dim_model)
        return emb
        
    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, t:torch.Tensor) -> torch.Tensor:
        return self.timembedding(t)
        
class DownSample(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 
                            kernel_size= 3,
                            stride= 2,
                            padding = 1)
        self.initialize()
        
    def initialize(self) -> None:
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return  self.conv(x)
    
class UpSample(nn.Module):
    def __init__(self, in_ch:int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
    
    def initialize(self) -> None:
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
        
class AttnBlock(nn.Module):
    def __init__(self, in_ch:int, n_group_norm :int = 32):
        super().__init__()
        self.group_norm = nn.GroupNorm(n_group_norm, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, 1, 0)
        
    def initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj, gain=1e-5)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)
        
        q = q.permute(0, 2, 3, 1).view(B, H*W, C)
        k = k.view(B, C, H*W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H*W, H*W]
        w = F.softmax(w, dim=-1)
        
        v = v.permute(0, 2, 3, 1).view(B, H*W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H*W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h
        
class ResBlock(nn.Module):
    def __init__(self, 
                in_ch:int,
                out_ch:int,
                tdim:int,
                dropout:float,
                n_group_norm:int = 32,
                is_attn:bool = False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(n_group_norm, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(n_group_norm, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
    
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, 1, 0)
        else :
            self.shortcut = nn.Identity()
            
        if is_attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        
    def initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)
        
    def forward(self, x:torch.Tensor, temb : torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        
        h = h + self.shortcut(x)
        return  self.attn(h)    