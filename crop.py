import numpy as np
import png
import math
from os import listdir
from os.path import isfile, join
import pandas as pd

x = (np.random.rand(5,4)*3400).astype(int)

def read_pixels(filename):
    w, h, pixels, _ = png.Reader(filename='../train/'+filename+'.png').read_flat()
    image = np.array(pixels).reshape(h,w)
    return image

def write_png(image, f, output_directory):
    with open(f'{output_directory}/{f}', 'wb') as png_file:
        image = np.uint8(image)
        h,w = image.shape
        w = png.Writer(w, h)
        w.write(png_file, image)


lsdir = listdir('./train/')
csv = pd.read_csv('./train.csv')
for f in lsdir:
    if isfile(join('./train/', f)):
        x = csv[csv['image_id'] == f.strip('.png')].values[:, 4:]
        print(x)
        if (np.vectorize(math.isnan)(x)).any():
            continue
        x = x.astype(int)
        print(x)
        # x_min = np.min(x[:,0])
        # y_min = np.min(x[:,1])
        # x_max = np.max(x[:,2])
        # y_max = np.max(x[:,3])
        # image = read_pixels()
        # h,w = image.shape
        # d = 10
        # image = image[min(y_min, max(0, y_min-d)) : max(y_max, min(y_max+d, h)), 
        #               min(x_min, min(0, x_min-d)) : max(x_max, min(x_max+d, w))]
        # write_png(image, f, './train')


