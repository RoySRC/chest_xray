import argparse
import os
import time
import random
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from DenseNet import DenseNet
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../DataLoaders')
from VinTrainDataLoader import VinTrainDataset


def time_string(time_seconds):
    h, m, s = [0] * 3
    h = time_seconds / 3600
    m = max(0, h - int(h)) * 60
    s = max(0, m - int(m)) * 60
    h, m, s = int(h), int(m), int(s)
    return f'{h:02d}:{m:02d}:{s:02d}'


def seed_everything(SEED=42):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True
seed_everything(64)



parser = argparse.ArgumentParser()
parser.add_argument("--csv",
                    nargs='?',
                    type=str,
                    default="/media/sajeeb/StorageDisk1/Data.csv")
parser.add_argument("--batch-size",
                    nargs='?',
                    type=int,
                    default=16)
parser.add_argument("--iterations",
                    nargs='?',
                    type=int,
                    default=8)
parser.add_argument("--test-split",
                    nargs='?',
                    type=float,
                    default=1)
parser.add_argument("--lr",
                    nargs='?',
                    type=float,
                    default=1e-4)
parser.add_argument("--weight-decay",
                    nargs='?',
                    type=float,
                    default=1e-7)
parser.add_argument("--image-source",
                    nargs='?',
                    type=str,
                    default="../images_416_enhanced/")
parser.add_argument("--model-file",
                    nargs='?',
                    type=str,
                    default="./pretrained_darknet_53.pt")
parser.add_argument("--num-dataloaders",
                    nargs='?',
                    type=int,
                    default=4)
parser.add_argument("--device",
                    nargs='?',
                    type=str,
                    default="cuda")
parser.add_argument("--print-model",
                    nargs='?',
                    type=bool,
                    default=False)
args = parser.parse_args()

model = DenseNet(15).to(args.device)
if args.print_model:
    print(model)
csv = pd.read_csv(args.csv)
iterations = args.iterations
batch_size = args.batch_size

# split the dataset
X = [f for f in listdir(args.image_source) if isfile(join(args.image_source, f))]
random.shuffle(X)
split = args.test_split / 100
xtrain, xtest = X[:int((1 - split) * len(X))], X[int((1 - split) * len(X)):]
train_dataset = VinTrainDataset(xtrain, csv, args.image_source, (416, 416), dtype=torch.float)
test_dataset = VinTrainDataset(xtest, csv, args.image_source, (416, 416), dtype=torch.float)
testloader = DataLoader(test_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=min(batch_size, args.num_dataloaders))
trainloader = DataLoader(train_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True,
                         num_workers=min(batch_size, args.num_dataloaders))
n_batches = int(len(train_dataset) / batch_size + 1)
n_val_batches = int(len(test_dataset) / batch_size + 1)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss = torch.nn.BCEWithLogitsLoss()

checkpoint_epoch = 0
if os.path.isfile(args.model_file):
    print("loading model file...")
    checkpoint = torch.load(args.model_file)
    checkpoint_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    try:
        print('pretrained epochs:', checkpoint_epoch,
              '| train loss:', f"{checkpoint['train_loss']:.5f}",
              '| val_loss:', f"{checkpoint['val_loss']:.5f}")
    except:
        pass
    for g in optimizer.param_groups:
        g['lr'] = args.lr
        g['weight_decay'] = args.weight_decay

else:
    print('No model file found.')

for epoc in range(checkpoint_epoch, iterations):
    losses = []
    start_time = time.time()
    # iterate through the batches
    model.train()
    for i, (x, y) in enumerate(trainloader):
        batch_start_time = time.time()
        optimizer.zero_grad()

        # with autocast():
        y_hat = model(x.to(args.device))
        l = loss(y_hat, y.to(args.device))
        losses.append(l.item())

        l.backward()
        optimizer.step()
        elapsed = time.time() - batch_start_time
        processed_images = (i + 1) * batch_size
        eta = time_string(elapsed * (n_batches - i - 1))
        batch_loss = losses[-1]
        print(f"\r loss on batch {i}: {batch_loss:.5f} |",
              processed_images, "| Iteration ETA:", eta, end='', flush=True)

    # validate
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, (x, y) in enumerate(testloader):
            batch_start_time = time.time()

            # with autocast():
            y_hat = model(x.to(args.device))
            l = loss(y_hat, y.to(args.device))
            val_loss.append(l.item())

            elapsed = time.time() - batch_start_time
            processed_images = (i + 1) * batch_size
            eta = time_string(elapsed * (n_val_batches - i - 1))
            val_batch_loss = val_loss[-1]
            print(f"\r loss on test batch {i}: {val_batch_loss:.5f} |",
                  processed_images, "| Iteration ETA:", eta,
                  end='', flush=True)
    val_loss = np.mean(val_loss)

    losses = np.mean(losses)
    elapsed = time.time() - start_time
    eta = time_string(elapsed * (iterations - epoc - 1))
    print('\r', (epoc + 1), " | train_loss:", f"{losses:.5f}",
          " | ETA:", eta, " | val_loss:", f"{val_loss:.5f}", flush=True)
    checkpoint_epoch += 1
    checkpoint = {
        'epoch': checkpoint_epoch,
        'train_loss': losses,
        'val_loss': val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, args.model_file)