import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
print(testloader)

def imshow(img):
    # unnormalize
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.subplot(121)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('random train image')
    plt.show()

# get random training images   
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

# print labels
file = open('./data/stl10_binary/class_names.txt', "r")
classes = file.read().split()

print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


## directory setting
if not os.path.exists('./data/train'):
    os.makedirs(os.path.join('./data/train'))

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
    np.save('./data/train/input_%d' % i, inputs)
    np.save('./data/train/label_%d' % i, labels)

if not os.path.exists('./data/test'):
    os.makedirs(os.path.join('./data/test'))

for i, data in enumerate(testloader, 0):
    inputs, labels = data
    np.save('./data/test/input_%d' % i, inputs)
    np.save('./data/test/label_%d' % i, labels)

print("Load Success")