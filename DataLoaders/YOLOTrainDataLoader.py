import numpy as np
import torch
from PIL import Image
from skimage import exposure
from torch.utils.data import Dataset


class YOLOTrainDataset(Dataset):
    def __init__(self, X, csv, image_source, image_shape, anchor_boxes, anchor_mask,
                 grid_sizes, dtype=torch.float16):
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
        self.anchor_boxes = anchor_boxes
        self.anchor_mask = anchor_mask
        self.grid_sizes = grid_sizes
        self.csv = csv
        self.max_h, self.max_w = image_shape

    def __len__(self):
        return (len(self.X))

    def __generate_target__(self, y_train):
        box_wh = y_train[:, 2:4] - y_train[:, 0:2]

        # compute the intersection of each box with each anchor
        bw = box_wh[:, 0].reshape(-1, 1)
        iw = np.minimum(np.tile(bw, [1, self.anchor_boxes.shape[0]]), self.anchor_boxes[:, 0])
        bh = box_wh[:, 1].reshape(-1, 1)
        ih = np.minimum(np.tile(bh, [1, self.anchor_boxes.shape[0]]), self.anchor_boxes[:, 1])
        intersection = iw * ih

        # Compute the union of each box with each anchor
        box_area = box_wh[:, 0] * box_wh[:, 1]
        box_area = np.tile(box_area.reshape(-1, 1), [1, 12])
        anchor_area = self.anchor_boxes[:, 0] * self.anchor_boxes[:, 1]
        union = box_area + anchor_area - intersection

        # compute the iou of each box with each anchor
        iou = intersection / union

        #
        anchor_idx = np.argmax(iou, axis=1)  # shape: (n_boxes,)
        y_train = np.concatenate([y_train, anchor_idx.reshape(-1, 1)], axis=1)
        # y_train: shape (n_boxes, len([xmin, ymin, xmax, ymax, class, anchor_index]))

        #
        y_outs = []
        for anchor_idxs, grid_size in zip(self.anchor_mask, self.grid_sizes):
            # 6 for xmin, ymin, xmax, ymax,objectness score, class
            y_true_out = np.zeros((grid_size, grid_size, anchor_idxs.shape[0], 4 + 1 + 14))
            for r in y_train:  # iterate through all boxes
                if np.isnan(r[0]) or np.isinf(r[0]):
                    continue
                anchor_eq = (anchor_idxs == int(r[-1]))
                if np.any(anchor_eq):
                    box = r[:4]
                    box_xy = (box[0:2] + box[2:4]) / 2  # box center
                    anchor_idx = np.where(anchor_eq)[0][0]
                    grid_x, grid_y = (box_xy // (1 / grid_size)).astype(int)
                    if (grid_x == 12):
                        print(box_xy)
                    class_probs = np.zeros(14)
                    indices = y_train[:, 4].astype(int)
                    indices = indices[indices != 14]
                    class_probs[indices] = 1.
                    class_probs = class_probs.tolist()
                    y_true_out[grid_x][grid_y][anchor_idx] = [box[0], box[1], box[2], box[3], 1] + class_probs

            y_outs.append(y_true_out)

        return y_outs

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

        pt /= self.max_h
        pl /= self.max_w
        print(pt, pl)
        y_train = self.csv[self.csv['image_id'] == filename.strip('.png')].values[:, [4, 5, 6, 7, 2]]
        y_train[:, 1] += pt
        y_train[:, 0] += pl
        y_train = y_train.astype(float)
        labels = self.__generate_target__(y_train)

        return image, labels
