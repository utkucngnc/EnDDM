from typing import List, Dict
import os
import json
import numpy as np
from PIL import Image

EXT = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']
MODIFICATIONS = ['blur', 'noise', 'rotate', 'flip', 'crop']


class datasetCreator:
    def __init__(self, args: Dict) -> None:
        self.input_path = args['input_path']
        self.mods = args['mods']
        if type(self.input_path) == list:
            self.fuse = True
            self.mod_1, self.mod_2 = self.mods
            assert all([os.path.exists(path) for path in self.input_path]), f"Path does not exist"
            if self.mod_1 is not None:
                assert all(mod in MODIFICATIONS for mod in self.mod_1), f"Mods must be one of {MODIFICATIONS}"
            if self.mod_2 is not None:
                assert all(mod in MODIFICATIONS for mod in self.mod_2), f"Mods must be one of {MODIFICATIONS}"
        else:
            self.fuse = False
            assert os.path.exists(self.input_path), f"Path {self.input_path} does not exist"
            if self.mods is not None:
                assert all(mod in MODIFICATIONS for mod in self.mods), f"Mods must be one of {MODIFICATIONS}"
        
        self.shape = args['shape']
        self.output_path = args['output_path']
        
        if self.fuse:
            from src.mmif_ddim.sample import fuse_imgs
            self.fuse_imgs = fuse_imgs

        self.__createFolders()
        self.__exportArgs(args)
        self.__createDataset()
        print(f"Dataset to {args['output_path']} created successfully")

    def __exportArgs(self, args: Dict) -> None:
        with open(f'{self.output_path}args.json', 'w') as f:
            json.dump(args, f)
    
    def __createFolders(self) -> None:
        if os.path.exists(self.output_path):
            import shutil
            print(f"Folder {self.output_path} already exists. Deleting it")
            shutil.rmtree(self.output_path)
            print(f"Folder {self.output_path} deleted")
        print(f"Creating folder {self.output_path}")
        os.mkdir(self.output_path)
            
    def __createDataset(self) -> None:
        if self.fuse:
            self.__createFusedDataset()
        else:
            self.__createSingleDataset()
    
    def __createFusedDataset(self) -> None:
        load_path_1, load_path_2 = self.input_path
        assert os.path.exists(load_path_1), f"Path {load_path_1} does not exist"
        assert os.path.exists(load_path_2), f"Path {load_path_2} does not exist"

        for f in os.listdir(load_path_1):
            if f.split('.')[-1] in EXT:
                assert f in os.listdir(load_path_2), f"File {f} not found in {load_path_2}"

                path_1 = load_path_1 + f
                path_2 = load_path_2 + f
                self.fuse_imgs(path_1, path_2, self.output_path)

    
    def __createSingleDataset(self) -> None:
        if self.input_path.split('.')[-1] in EXT:
            self.__processTiffImage()
        else:
            for f in os.listdir(self.input_path): # add channel check assertion
                if f.split('.')[-1] in EXT:
                    img = np.array(Image.open(self.input_path + f))
                    img = self.__applyMod(img, self.mods)
                    img.save(self.output_path + f)
    
    def __processTiffImage(self,
                           portion: float = 1.0) -> None:
        
        from skimage import io

        tiff_img = io.imread(self.input_path) # (B x H x W)
        if portion != 1.0:
            indices = np.random.randint(tiff_img.shape[0], size=int(tiff_img.shape[0] * portion))
        else:
            indices = np.arange(tiff_img.shape[0])
        for i in indices:
            img = np.expand_dims(tiff_img[i], axis = -1)
            img = np.repeat(img, self.shape[-1], axis = -1)
            img = self.__resize(img)
            img = self.__applyMod(img, self.mods)
            img.save(self.output_path + f'{i}.png')
    
    def __applyMod(self, img: Image, mods: List[str] or None) -> Image:
        if mods is None:
            return Image.fromarray(img)
        else:
            for mod in mods:
                if mod == 'blur':
                    img = self.__blur(img)
                elif mod == 'noise':
                    img = self.__noise(img)
                elif mod == 'rotate':
                    img = self.__rotate(img)
                elif mod == 'flip':
                    img = self.__flip(img)
                elif mod == 'crop':
                    img = self.__crop(img)
                else:
                    raise ValueError(f"Mod {mod} not recognized")
            return Image.fromarray(img)
    
    def __blur(self, img: np.ndarray) -> np.ndarray:
        idx = np.arange(1,img.shape[1]-1,dtype=int)
        img[:,idx][idx,:][:] = (5*img[:,idx][idx,:][:] + img[:,idx-1][idx,:][:] + img[:,idx+1][idx,:][:] + img[:,idx][idx-1,:][:] + img[:,idx][idx+1,:][:])
        return np.clip(img,0,255)
    
    def __noise(self, img: np.ndarray) -> np.ndarray:
        return np.clip(img + np.random.randint(0,255,img.shape),0,255).astype(np.uint8)

    def __rotate(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.rot90(img),0,255).astype(np.uint8)
    
    def __flip(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.flipud(img),0,255).astype(np.uint8)
    
    def __crop(self, img: np.ndarray) -> np.ndarray:
        roi_x = sorted(np.random.randint(0, img.shape[0],2))
        roi_y = sorted(np.random.randint(0, img.shape[1],2))
        img[roi_x[0]:roi_x[1],roi_y[0]:roi_y[1],:] = 0
        return np.clip(img,0,255).astype(np.uint8)
    
    def __resize(self, img: np.ndarray) -> np.ndarray:
        return np.clip(np.array(Image.fromarray(img).resize(self.shape[:2])),0,255).astype(np.uint8)