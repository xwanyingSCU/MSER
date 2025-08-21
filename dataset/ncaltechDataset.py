import numpy as np
import torch
from torch.utils.data import Dataset,random_split
import h5py
import torch.nn as nn
class caltech101dataset(Dataset):
    def __init__(self, data_path, key,class_list):
        self.data = h5py.File(data_path, "r")
        self.class_list = class_list
        self.keys=[item for sublist in key for item in sublist]

    def __len__(self):
        return self.keys.__len__()

    def __getitem__(self, index):
        org_data = self.data.get(self.keys[index])[:]
        org_data = torch.tensor(org_data)
        org_data = org_data.transpose(0, 2).float()
        org_data = org_data.unsqueeze(0)
        expanded_image = nn.functional.interpolate(org_data, [224, 224])
        expanded_image = expanded_image.squeeze(0)
        ch_label = self.keys[index].split('-')[0]
        label = self.class_list.index(ch_label)
        if label < 0 or label > 100:
            print("label error")
        return expanded_image, label