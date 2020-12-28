import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from model import *
from data_loader import *
from util import *

## Parser
parser = argparse.ArgumentParser(description="Train the VGGNet",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.00001, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./data", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## training parameter setting
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

file = open('./data/stl10_binary/class_names.txt', "r")
classes = file.read().split()

print(classes)


## Train
if mode == 'train':
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

    num_data_train = len(dataset_train)

    num_batch_train = np.ceil(num_data_train / batch_size)
else:
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'))
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

# network load
VGG_types = {
    # VGG11
    "VGG-A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # VGG11 LRN
    "VGG-A-LRN": [64, "L", "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # VGG13
    "VGG-B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    # VGG16
    "VGG-D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    # VGG19
    "VGG-E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

net = VGGNet(in_channels=3, num_classes=len(classes), VGG_type=VGG_types['VGG-D']).to(device)

# loss function
fn_loss = nn.CrossEntropyLoss().to(device)

# optimizer
optim = torch.optim.Adam(net.parameters(),lr=lr)

st_epoch = 0

# TRAIN MODE
if mode == 'train':
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
        
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # forward pass
            label = data['label'].to(device)
            input = data['input'].to(device)

            label = torch.squeeze(label)
            input = torch.squeeze(input)

            output = net(input)

            # backward pass
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # loss function
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))


            if epoch % 10 == 0:
                save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    loss_arr = []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        loss = fn_loss(output, label)

        print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
                        (batch, num_batch_test, np.mean(loss_arr)))