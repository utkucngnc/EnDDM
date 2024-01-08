################################################################
# The dataset will be used to fine-tune the upsampling network #
################################################################

import torch as th
import os
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Any

IMG_EXT = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']
RES_TYPE = ['low_res', 'high_res']
MODE = ['train', 'test']
ROOT = '.data/'

class BatteryDataset(Dataset):
    def __init__(self, mode: str, res_type: str):
        super().__init__()
        assert res_type in RES_TYPE
        assert mode in MODE
        self.root = f'{ROOT}{mode}/{res_type}'
        self.imgs = []
        self.labels = []
        self.__load(self.root)
    
    def __getitem__(self, index) -> List[Any]:
        if th.is_tensor(index):
            index = index.tolist()
        return Image.open(self.imgs[index]), self.labels[index]
    
    def __len__(self) -> int:
        return len(self.imgs)
    
    def __load(self, path: str = None) -> None:
        if not os.path.exists(path):
            raise ValueError("Path does not exist")
        if not os.path.isdir(path):
            raise ValueError("Path is not a directory")
        for file in os.listdir(path):
            if file.split('.')[-1] in IMG_EXT:
                self.imgs.append(os.path.join(path, file))
                self.labels.append(int(file.split('.')[0]))
        self.imgs.sort()
        self.labels.sort()