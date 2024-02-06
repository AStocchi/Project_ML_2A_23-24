import copy
import torch
import matplotlib.pyplot as plt

from scipy import ndimage
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from client import Client

from utils.utils import HardNegativeMining, MeanReduction


class Client_rot_dataset(Client):
    def __init__(self, args, dataset, model, test_client=False, group=0):
        super(Client_rot_dataset, self).__init__(args, dataset, model, test_client)

        self.group = group

    def rotate_dataset(self, vect_images):
        if self.group != 0:
            list_groups = [0, 15, 30, 45, 60, 75]
            for num, sample in enumerate(vect_images):
                sample = sample.permute(1, 2, 0)
                vect_images[num] = torch.tensor(ndimage.rotate(sample, list_groups[self.group], reshape=False)).permute(
                    2, 0, 1)
        return vect_images

    def transform_image(self, y, dim_y): #ovveride transform method
        z = torch.reshape(y, (dim_y, 1, 28, 28))
        z_rot = self.rotate_dataset(z)
        return z_rot
