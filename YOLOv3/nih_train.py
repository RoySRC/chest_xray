import argparse
import os
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from nih_dataloader import NIHImageDataset
from nih_val_dataloader import NIHValDataset
from DarkNet import DarkNet_53

def time_string(time_seconds):
    h,m,s = [0]*3
    h = time_seconds / 3600
    m = max(0, h - int(h)) * 60
    s = max(0, m - int(m)) * 60
    h, m, s = int(h), int(m), int(s)
    return f'{h:02d}:{m:02d}:{s:02d}'

def generate_labels(csv):
    """
    :param csv: pandas csv
    :return: all unique labels for images
    """
    l = []
    for f in csv['Finding Labels']:
        l += f.split('|')
    l = sorted(list(np.unique(l)))
    return l


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
args = parser.parse_args()

model = DarkNet_53([1, 2, 8, 8, 4], (416, 416)).to(args.device)
csv = pd.read_csv(args.csv)
class_labels = {l:i for i,l in enumerate(generate_labels(csv))}
iterations = args.iterations
batch_size = args.batch_size

# split the dataset
X = list(csv['Image Index'].values)
random.shuffle(X)
split = 2/100
xtrain, xtest = X[:int((1-split)*len(X))], X[int((1-split)*len(X)):]
image_source = args.image_source
train_dataset = NIHImageDataset(xtrain, csv, class_labels, image_source)
test_dataset = NIHValDataset(xtest, csv, class_labels, image_source)
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
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss = torch.nn.BCEWithLogitsLoss()

checkpoint_epoch = 0
if os.path.isfile(args.model_file):
    print("loading model file...")
    print('pretrained epochs:', checkpoint_epoch)
    checkpoint = torch.load(args.model_file)
    checkpoint_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    for g in optimizer.param_groups:
        g['lr'] = 1e-2
else:
    print('No model file found.')

for epoc in range(iterations):
    losses = []
    checkpoint_epoch += 1
    start_time = time.time()
    # iterate through the batches
    model.train()
    for i, (x, y) in enumerate(trainloader):
        batch_start_time = time.time()
        optimizer.zero_grad()

        with autocast():
            y_hat = model(x.to(args.device))
            l = loss(y_hat, y.to(args.device))
            losses.append(l.item())

        l.backward()
        optimizer.step()
        elapsed = time.time()-batch_start_time
        processed_images = (i+1)*batch_size
        eta = time_string(elapsed * (n_batches - i - 1))
        batch_loss = np.round(losses[-1], 2)
        print(f"\r loss on batch {i}:", batch_loss, '|', 
              processed_images, "| Iteration ETA:", eta, end='', flush=True)

    # validate
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i,(x,y) in enumerate(testloader):
            batch_start_time = time.time()

            with autocast():
                y_hat = model(x.to(args.device))
                l = loss(y_hat, y.to(args.device))
                val_loss.append(l.item())
            
            elapsed = time.time() - batch_start_time
            processed_images = (i+1)*batch_size
            eta = time_string(elapsed * (n_val_batches - i - 1))
            val_batch_loss = np.round(val_loss[-1], 2)
            print(f"\r loss on test batch {i}:", val_batch_loss, '|', 
                  processed_images, "| Iteration ETA:", eta,
                  end='', flush=True)
    val_loss = np.round(np.mean(val_loss), 5)

    losses = np.round(np.mean(losses), 5)
    elapsed = time.time() - start_time
    eta = time_string(elapsed * (iterations - epoc - 1))
    print('\r', (epoc + 1), " | train_loss:", losses, 
          " | ETA:", eta, " | val_loss:", val_loss, flush=True)
    checkpoint = {
        'epoch': checkpoint_epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, args.model_file)