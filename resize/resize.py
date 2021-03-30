import argparse

import numpy as np
import psutil
import os
from os import listdir
from os.path import isfile, join
import time


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input directory containing images')
parser.add_argument('--output',type=str,  help='output directory to store images to')
parser.add_argument('--output-size', type=int, help='the output image resolution')
args = parser.parse_args()


files = [f for f in listdir(args.input) if isfile(join(args.input, f))]
nProcesses = 12
process_ids = []
chunk_size = int(np.ceil(len(files) / nProcesses))

start = 0
for i in range(nProcesses):
    end = (start + chunk_size) if i + 1 != nProcesses else len(files)
    pid = os.fork()
    if pid == 0:  # child process
        os.execlp('python3', 'python3', 'resize_child_process.py',
                  str(i),
                  str(nProcesses),
                  str(args.input),
                  str(args.output),
                  str(args.output_size))
    else:
        start = end
        process_ids.append(pid)

print(f'started processes: {str(process_ids)}')

while len(process_ids) != 0:
    for p in process_ids:
        try:
            if psutil.Process(p).status() == 'zombie':
                os.waitpid(p, 0)
                process_ids.remove(p)
        except:
            process_ids.remove(p)
    time.sleep(0.5)

print("Done!")

