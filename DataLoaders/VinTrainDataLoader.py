import numpy as np
import torch
from PIL import Image
from skimage import exposure
from torch.utils.data import Dataset


class VinTrainDataset(Dataset):
    def __init__(self, X, csv, image_source, image_shape, dtype=torch.float16):
        '''
        dataset used by dataloader.
        :param X: The inputs to the model. These are the names of the images as they appear in the filesystem
        :param csv: pandas dataframe of the loaded csv
        :param image_source: URL to the source folder containing the images
        :param image_shape: height and width of the image
        :param dtype: torch float datatype. Default is torch.float16
        '''
        self.dtype = dtype
        self.X = X
        self.PREFIX = image_source
        self.csv = csv
        self.max_h, self.max_w = image_shape

    def __len__(self):
        return (len(self.X))

    def __pad__(self, array):
        h, w = array.shape
        pt = (self.max_h - h) // 2  # padding top
        pb = self.max_h - pt - h  # padding bottom
        pl = (self.max_w - w) // 2  # padding left
        pr = self.max_w - pl - w  # padding right
        return np.pad(array, pad_width=((pt, pb), (pl, pr)), constant_values=((0, 0), (0, 0))), pt, pl

    def __getitem__(self, i):
        filename = self.X[i]

        image = np.array(Image.open(f'{self.PREFIX}/{filename}').convert('L'))
        image, pt, pl = self.__pad__(image)
        assert (image.shape == (self.max_h, self.max_w)), \
            f"Expected image shape after padding to be of height {self.max_h} and width {self.max_w}"
        image = exposure.equalize_adapthist(image)
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image / 255.0, dtype=self.dtype)

        l = self.csv[self.csv['image_id'] == filename.strip('.png')]
        l = np.unique(l['class_id'].values)
        labels = torch.zeros(15, dtype=self.dtype)
        labels[l] = 1.0 # set the values of the indexes at numeric class labels to 1.0

        return image, labels