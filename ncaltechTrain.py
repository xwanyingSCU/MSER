import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import h5py
import os

import torch
import argparse
import sys
import torch.nn.functional as F
sys.path.append("..")

from model.ncaltech  import FPN_Multi_ResNet34_Moe


from dataset.ncaltechDataset import caltech101dataset
import numpy as np
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_random_seed(42)
learning_rate = 0.0001

parser = argparse.ArgumentParser()
# parser.add_argument("--name", required=True, help="The path of dataset")
parser.add_argument(
    "--img_type", default='log', help="The encoding way"
)
parser.add_argument("--epoch", default=101, type=int, help="The number of epochs")
parser.add_argument("--batch_size", default=16, type=int, help="batch size")
parser.add_argument("--cuda", default="2", help="The GPU ID")
args = parser.parse_args()
img_type_dict={
    'log':{'channel_num':6, 'file': "/data/xwy/ncaltech/feature_log_win3.hdf5"},
}
device = torch.device("cuda:"+args.cuda if torch.cuda.is_available() else "cpu")



load_path = '/data/xwy/ncaltech/ncaltech_label.npy'
folder_name = np.load(load_path, allow_pickle=True)
folder_names = folder_name.tolist()


model =FPN_Multi_ResNet34_Moe(101)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


train_data =[]
test_data=[]
data = h5py.File(img_type_dict[args.img_type]['file'], "r")
folder_names_sorted = sorted(folder_names, key=len, reverse=True)
remaining_keys = set(data.keys())
for label in folder_names_sorted:

    keys = [key for key in remaining_keys if key.startswith(label)]
    remaining_keys -= set(keys)
    random.shuffle(keys)

    train_keys=keys[:int(0.8 * len(keys))]
    test_keys=keys[int(0.8 * len(keys)):]
    test_data.append(test_keys)
    train_data.append(train_keys)

train_dataset =  caltech101dataset(img_type_dict[args.img_type]['file'],train_data,folder_names_sorted)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)

test_dataset =  caltech101dataset(img_type_dict[args.img_type]['file'], test_data,folder_names_sorted)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False)

old_accuracy = 0
old_epoch=0
for epoch in range(1, args.epoch):
    correct = 0
    total = 0

    model.train()
    for i, data in enumerate(tqdm(train_dataloader, desc="Epoch: {}".format(epoch))):

        input, label = data
        input = input.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        end_point = model(input)
        loss = F.cross_entropy(end_point, label)
        pred = end_point.max(1)[1]
        total += len(label)
        correct += pred.eq(label).sum().item()
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for index, data in enumerate(tqdm(test_dataloader, desc="Epoch: {}".format(epoch))):
            input, label = data
            input = input.to(device)
            label = label.to(device)
            end_point = model(input)
            pred = end_point.max(1)[1]
            total += len(label)
            correct += pred.eq(label).sum().item()

        accuracy = correct / total
        if old_accuracy<accuracy:
            old_accuracy=accuracy
            old_epoch=epoch
        print("test acc is {}".format(correct / total))
        print("best acc is {}".format(old_accuracy), "   epoch is ", old_epoch)

