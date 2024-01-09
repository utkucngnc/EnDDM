from functools import partial
import os
import torch
from .guided_diffusion.unet import create_model
from .guided_diffusion.gaussian_diffusion import create_sampler
from .util.logger import get_logger
import cv2
import numpy as np
from skimage.io import imsave
import warnings

from .config import diffusion_config, model_config
warnings.filterwarnings('ignore')

def image_read(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def fuse_imgs(path_1, path_2, output_path):
   
    # logger
    logger = get_logger('DPS')
    
    # Device setting
    device_str = f"cuda:0" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)
   
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

  
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model)
   
    # Working directory
    img_name = path_1.split('/')[-1]
    os.makedirs(output_path, exist_ok=True)
    for img_dir in ['recon', 'progress']:
        os.makedirs(os.path.join(output_path, img_dir), exist_ok=True)

    img_1 = image_read(path_1,mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 
    img_2 = image_read(path_2, mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0 

    img_1 = img_1*2-1
    img_2 = img_2*2-1

    # crop to make divisible
    scale = 32
    h, w = img_1.shape[2:]
    h = h - h % scale
    w = w - w % scale

    img_1 = ((torch.FloatTensor(img_1))[:,:,:h,:w]).to(device)
    img_2 = ((torch.FloatTensor(img_2))[:,:,:h,:w]).to(device)
    assert img_1.shape == img_2.shape

    logger.info(f"Inference for image {img_name}")

    # Sampling
    seed = 3407
    torch.manual_seed(seed)
    x_start = torch.randn((img_1.repeat(1, 3, 1, 1)).shape, device=device)  

    with torch.no_grad():
        sample = sample_fn(x_start=x_start, record=True, I = img_1, V = img_2, save_root=output_path, img_index = os.path.splitext(img_name)[0], lamb=0.5,rho=0.001)

    sample= sample.detach().cpu().squeeze().numpy()
    sample=np.transpose(sample, (1,2,0))
    sample=cv2.cvtColor(sample,cv2.COLOR_RGB2YCrCb)[:,:,0]
    sample=(sample-np.min(sample))/(np.max(sample)-np.min(sample))
    sample=((sample)*255).astype(np.uint8)
    imsave(os.path.join(os.path.join(output_path, 'recon'), "{}.png".format(img_name.split(".")[0])),sample)
