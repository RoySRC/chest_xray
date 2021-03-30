import argparse

import psutil
import os
from os import listdir
from os.path import isfile, join
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input directory containing images')
parser.add_argument('--output',type=str,  help='output directory to store inverted images to')
args = parser.parse_args()

files = [f for f in listdir(args.input) if isfile(join(args.input, f))]
nProcesses = 12
process_ids = []
chunk_size = int(np.ceil(len(files) / nProcesses))

for i in range(nProcesses):
    pid = os.fork()
    if pid == 0:  # child process
        os.execlp('python3', 'python3', 'inverter_process.py',
                  str(i), # process id
                  str(nProcesses),
                  args.input,
                  args.output)
    else:
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
