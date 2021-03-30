from torch.utils.data import Dataset
import numpy as np
import png
import torch

class DarkNetImageDataset(Dataset):
    def __init__(self, path, csv, image_dims):
        self.X = path
        self.csv = csv
        self.max_h, self.max_w = image_dims

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

        w, h, pixels, _ = png.Reader(filename=f'../train/{filename}.png').read_flat()
        image = np.array(pixels).reshape(h, w)
        #         image = exposure.equalize_adapthist(image)
        image, pt, pl = self.__pad__(image)
        assert (image.shape == (self.max_h, self.max_w)) ,\
                f"Expected image shape after padding to be of height {self.max_h} and width {self.max_w}"
        image = np.expand_dims(image, axis=0)
        image = torch.tensor(image / 255.0, dtype=torch.float16)

        csv = self.csv[self.csv['image_id'] == filename]
        class_ids = csv['class_id'].values

        bboxes = csv.values[:, 4:]
        if np.isnan(np.sum(bboxes)):
            bboxes[:, [0, 1]] = 0.0
            bboxes[:, [2, 3]] = 1.0
        else:
            bboxes = np.vectorize(lambda x: int(x))(bboxes)
            bboxes[:, 0::2] = (bboxes[:, 0::2] + pl) / w  # normalize the x coordinates
            bboxes[:, 1::2] = (bboxes[:, 1::2] + pt) / h  # normalize the y coordinates

        output_bboxes = np.zeros(14 * 4)
        if (class_ids == 14).any():  # if there is no abnormality
            lbl = np.zeros(15)
        else:
            l = np.zeros(15)
            l[0] = 1.  # there is an object of interest
            l[class_ids] = 1.  # set the expected probabilities to 1.0
            lbl = l
            for i, c in enumerate(class_ids):
                output_bboxes[c * 4:(c + 1) * 4] = bboxes[i]

        probs_bboxes = np.hstack((lbl, output_bboxes)).astype(np.float16)
        #         probs = torch.tensor(lbl, dtype=torch.long)
        #         bboxes = torch.tensor(bboxes, dtype=torch.float)
        probs_bboxes = torch.tensor(probs_bboxes, dtype=torch.float16)
        return image, probs_bboxes