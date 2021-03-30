from YOLOLoss import YOLOLoss
from yolov3 import YOLOv3

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

import sys
sys.path.insert(1, '../DataLoaders')
from YOLOTrainDataLoader import YOLOTrainDataset




def time_string(time_seconds):
    h, m, s = [0] * 3
    h = time_seconds / 3600
    m = max(0, h - int(h)) * 60
    s = max(0, m - int(m)) * 60
    h, m, s = int(h), int(m), int(s)
    return f'{h:02d}:{m:02d}:{s:02d}'

def transform(y_hat):
    """
    Transform the output of the model to contain the correct dimension
    :param y_hat: The predicted model output
    :return: a tuple containing the corrected shapes of all the elements in y_hat
    """
    large, medium, small = y_hat
    _bs = large.shape[0]

    # (batch, channel, h, w) -> (batch, w, h, channel)
    large = torch.transpose(large, 1, 3)
    medium = torch.transpose(medium, 1, 3)
    small = torch.transpose(small, 1, 3)

    # (batch, w, h, channel) -> (batch, w, h, anchors, 4+1+n_classes)
    w, h = large.shape[1:3]
    large = large.view(_bs, w, h, 4, -1)
    w, h = medium.shape[1:3]
    medium = medium.view(_bs, w, h, 4, -1)
    w, h = small.shape[1:3]
    small = small.view(_bs, w, h, 4, -1)

    return large, medium, small

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
                    default=8)
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
parser.add_argument("--backbone-model-file",
                    nargs='?',
                    type=str,
                    default="./pretrained_densenet_53.pt")
parser.add_argument("--model-file",
                    nargs='?',
                    type=str,
                    default="./pretrained_densenet_53.pt")
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

#######################################################################################

anchor_boxes = np.array([
    [ 16.18261013,  17.15899729],
    [ 36.3122743 ,  41.61847269],
    [ 45.47505783,  18.36459174],
    [ 48.72486224,  63.21332737],
    [ 54.45729627, 125.53348132],
    [ 76.31485206,  46.19644887],
    [ 81.95501274,  88.50688023],
    [ 92.96169628, 196.46041862],
    [ 99.68017319, 133.02863533],
    [127.71848912, 266.96994899],
    [144.06471319,  52.38340542],
    [182.76204773,  69.06707947]
]) / 416

anchor_mask = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10 ,11]
])


model = YOLOv3([128, 640, 640], 15, args.backbone_model_file).to(args.device)
optimizer = optim.Adam(model.yolo_head.parameters(), lr=args.lr)
loss_fns = [YOLOLoss(anchor_boxes[mask], 14) for mask in anchor_mask]

csv = pd.read_csv(args.csv)
# csv = csv[csv['x_min'].notna()]
iterations = args.iterations
batch_size = args.batch_size


X = [f for f in listdir(args.image_source) if isfile(join(args.image_source, f))]
random.shuffle(X)
split = args.test_split / 100
xtrain, xtest = X[:int((1 - split) * len(X))], X[int((1 - split) * len(X)):]
train_dataset = YOLOTrainDataset(xtrain, csv, args.image_source, (416, 416), anchor_boxes,
                                 anchor_mask, [12, 25, 51], dtype=torch.float)
test_dataset = YOLOTrainDataset(xtest, csv, args.image_source, (416, 416), anchor_boxes,
                                anchor_mask, [12, 25, 51], dtype=torch.float)
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

# load any previously saved model
checkpoint_epoch = 0
if os.path.isfile(args.model_file):
    print("loading model file...")
    checkpoint = torch.load(args.model_file)
    checkpoint_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('pretrained epochs:', checkpoint_epoch,
          '| train loss:', f"{checkpoint['train_loss']:.5f}",
          '| val_loss:', f"{checkpoint['val_loss']:.5f}")
    for g in optimizer.param_groups:
        g['lr'] = args.lr
        g['weight_decay'] = args.weight_decay
else:
    print('No model file found.')

# Start model training
for epoc in range(checkpoint_epoch, iterations):
    losses = []
    start_time = time.time()
    # iterate through the batches
    model.train()
    for i, (x, y) in enumerate(trainloader):
        batch_start_time = time.time()
        optimizer.zero_grad()

        # with autocast():
        y_hat = transform(model(x.to(args.device)))
        detection_head_losses = []
        for prediction, target, loss_fn in zip(y_hat, y, loss_fns):
            detection_head_losses.append(loss_fn(prediction, target.to(args.device)))

        detection_head_losses = torch.cat(detection_head_losses)
        sum_detection_head_losses = torch.sum(detection_head_losses)
        sum_detection_head_losses.requires_grad = True
        losses.append(sum_detection_head_losses.item())
        sum_detection_head_losses.backward()
        optimizer.step()

        # Print statistics
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
            y_hat = transform(model(x.to(args.device)))
            detection_head_losses = []
            for prediction, target, loss_fn in zip(y_hat, y, loss_fns):
                detection_head_losses.append(loss_fn(prediction, target.to(args.device)))

            val_loss.append(sum(detection_head_losses).item())

            # Print statistics
            elapsed = time.time() - batch_start_time
            processed_images = (i + 1) * batch_size
            eta = time_string(elapsed * (n_val_batches - i - 1))
            print(f"\r loss on test batch {i}: {val_loss[-1]:.5f} |",
                  processed_images, "| Iteration ETA:", eta,
                  end='', flush=True)

    # Print iteration statistics
    val_loss = np.mean(val_loss)
    losses = np.mean(losses)
    elapsed = time.time() - start_time
    eta = time_string(elapsed * (iterations - epoc - 1))
    print('\r', (epoc + 1), " | train_loss:", f"{losses:.5f}",
          " | ETA:", eta, " | val_loss:", f"{val_loss:.5f}", flush=True)

    # Save the currently trained model
    checkpoint_epoch += 1
    checkpoint = {
        'epoch': checkpoint_epoch,
        'train_loss': losses,
        'val_loss': val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, args.model_file)










