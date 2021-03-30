from skimage import exposure
import os, png
from os import listdir
from os.path import join, isfile

import skimage
import numpy as np
from PIL import Image
import sys


pid, nProcesses, in_dir, out_dir = sys.argv[1:]
pid, nProcesses = int(pid), int(nProcesses)
files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

for i,f in enumerate(files):
    if i % nProcesses == pid:
        img = Image.open(f'{in_dir}/{f}').convert('L')  # image extension *.png,*.jpg
        img = skimage.util.img_as_ubyte(exposure.equalize_adapthist(np.array(img)))
        img = Image.fromarray(img)
        img.save(f'{out_dir}/{f}')
        # shape = img.shape
        # with open(f'{out_dir}/{f}', 'wb') as png_file:
        #     png.Writer(shape[1], shape[0], bitdepth=8).write(png_file, img.tolist())

print(f'done with {os.getpid()}')

