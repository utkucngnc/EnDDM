from skimage import io
import numpy as np
import cv2

def noise_like(shape, noise_type='gaussian'):
    if noise_type == 'gaussian':
        return np.random.randn(*shape).astype('float32')
    elif noise_type == 'uniform':
        return np.random.uniform(-1, 1, shape).astype('float32')
    else:
        raise NotImplementedError
    
def tiff_read(path):
    img = io.imread(path)
    return img

def image_read(path, mode='RGB'):
    img_BGR = io.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def image_write(path, img, mode='GRAY'):
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == 'GRAY':
        img = np.round(img)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(path, img)

def image_show(img, mode='GRAY'):
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif mode == 'GRAY':
        img = np.round(img)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(img, size):
    return cv2.resize(img, size)