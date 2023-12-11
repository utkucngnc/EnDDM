from utils import *
import cv2

def fill_input_file():
    shape = (512,512)
    noised_img = noise_like(shape, noise_type='gaussian')

    images = tiff_read('../Pristine/PTY_pristine_raw.tif')
    path_ir = './input/ir/'
    path_vi = './input/vi/'

    for index in range(0,len(images),100):
        print(f'Saving image: {index}')
        img = cv2.resize(images[index], shape)
        image_write(f'{path_ir}{index}.jpg', img, mode='GRAY')
        image_write(f'{path_vi}{index}.jpg', noised_img, mode='GRAY')

if __name__ == '__main__':
    fill_input_file()