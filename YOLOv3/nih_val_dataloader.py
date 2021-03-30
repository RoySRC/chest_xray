from skimage import exposure
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image


class NIHValDataset(Dataset):
    def __init__(self, X, csv, class_labels, image_source):
        self.X = X
        self.PREFIX = image_source
        self.csv = csv
        self.class_labels_fn = np.vectorize(class_labels.get)

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        dtype = torch.float16
        filename = self.X[i]

        image = np.array(Image.open(f'{self.PREFIX}/{filename}').convert('L'))
        image = exposure.equalize_adapthist(image)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image / 255.0, dtype=dtype)

        csv = self.csv[self.csv['Image Index'] == filename]
        l = csv['Finding Labels'].values[0].split('|') # targets in csv
        l = self.class_labels_fn(l) # find numeric class labels
        labels = torch.zeros(15, dtype=dtype)
        labels[l] = 1.0 # set the values of the indexes at numeric class labels to 1.0

        return image, labels