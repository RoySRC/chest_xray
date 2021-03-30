import os
from os import listdir
from os.path import join, isfile

import numpy as np
from PIL import Image
import sys


pid, nProcesses, in_dir, out_dir, size = sys.argv[1:]
pid, nProcesses, size = int(pid), int(nProcesses), int(size)
files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]

for i,f in enumerate(files):
    if i % nProcesses == pid:
        try:
            img = Image.open(f'{in_dir}/{f}').convert('L') # image extension *.png,*.jpg
            h,w = np.array(img).shape
            # print(np.array(img).shape, end='                  \r')
        except Exception as e:
            print(e)
            print(np.array(img).shape)
            print(f)
            continue

        if h<=size and w<=size:
            continue

        if w <= h:
            new_height = size
            new_width = int(np.ceil(w/h*new_height))
            assert new_width <= new_height, \
                f"width {new_width} should be less that or equal to {new_height} for {f}"
        else:
            new_width = size
            new_height = int(np.ceil(h/w*new_width))
            assert new_height <= new_width, \
                f"height {new_height} should be less that or equal to {new_width} for {f}"

        img = img.resize((new_width, new_height), Image.ANTIALIAS)
        img.save(f'{out_dir}/{f}')


print(f'done with {os.getpid()}')