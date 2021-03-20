from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as T

from PIL import Image
from os import listdir, path
from typing import List

FILE_NAME = {128:'128x128',
             256:'256x256',
             512:'512x512',
             1024:'1024x1024'}

class celebA_HQ_DataModule(LightningDataModule):
    '''
    celebA HQ Dataset,
    - image dataset path.
    - celebA-HQ
        - 128x128 
        - 256x256
        - 512x512
        - 1024x1024
    '''
    def __init__(self,
                data_size : int = 128,
                data_path : str = None,
                test_batch_size : int = 32,
                train_batch_size : int = 32,
                test_ratio : float = 0.2,
                num_workers: int = 12,
                ):
        super().__init__()
        assert data_size in [128, 256, 512, 1024], 'data size should be one of 128, 256, 512, 1024'
        self.data_size = data_size
        self.data_path = data_path
        self.sub_path = FILE_NAME[data_size]
        self.test_batch_size = test_batch_size
        self.train_batch_size = train_batch_size
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        
    def setup(self, stage:str = None):
        data_num = len(listdir(path.join(self.data_path, self.sub_path)))
        self.train_dataset = celebA_HQ(data_path=path.join(self.data_path, self.sub_path), data_size=self.data_size, idx_list=[int(self.test_ratio*data_num), None])
        self.val_dataset = celebA_HQ(data_path=path.join(self.data_path, self.sub_path), data_size=self.data_size, idx_list=[None, int(self.test_ratio*data_num)])
        print('number of data, train : {}, val : {}'.format(len(self.train_dataset), len(self.val_dataset)))
    
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, pin_memory=True, num_workers=self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_batch_size, pin_memory=True, num_workers=self.num_workers)
        
    
    
class celebA_HQ(Dataset):
    def __init__(self,
                data_path : str = None,
                data_size : int= 128,
                idx_list : List[int] = None):
        super().__init__()
        self.data_path = data_path
        self.data_list = listdir(data_path)
        if idx_list is not None:
            self.data_list = self.data_list[idx_list[0] : idx_list[1]]
        self.data_size = data_size
        self.transforms = T.Compose([T.Resize((self.data_size, self.data_size)),
                                    T.RandomHorizontalFlip(),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))])
        
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img = Image.open(path.join(self.data_path, self.data_list[idx]))
        return self.transforms(img)
