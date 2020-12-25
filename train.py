import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from model import *
from data_loader import *
from util import *

file = open('./data/stl10_binary/class_names.txt', "r")
classes = file.read().split()

print(classes)

data_dir = './data'
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Train
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

num_data_train = len(dataset_train)

num_batch_train = np.ceil(num_data_train / batch_size)

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
optim = torch.optim.Adam(net.parameters(),lr=0.00001)

st_epoch = 0
num_epoch = 50
ckpt_dir = './checkpoint'

for epoch in range(st_epoch + 1, num_epoch + 1):
    net.train()
    loss_arr = []

    for i, data in enumerate(loader_train, 1):
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
                  (epoch, num_epoch, i, num_batch_train, np.mean(loss_arr)))


        if epoch % 10 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)