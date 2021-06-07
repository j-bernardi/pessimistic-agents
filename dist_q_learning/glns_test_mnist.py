
import glns
import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose(
    [transforms.ToTensor(),transforms.Normalize((0.5,),(1,))])

# transform = transforms.Compose(
#     [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='/home/peter/Documents/ML/mnist_ae/data', train=True,
                                        download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.MNIST(root='/home/peter/Documents/ML/mnist_ae/data', train=False,
                                       download=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

ggln1 = glns.GGLN([8,8,8,1], 28*28, 8, lr=3e-2, min_sigma_sq=0.05, bias_len=3,
                init_bias_weights=[None, None, None])

glns_set = []

ii = 0
for data in trainset:
    if ii%100==0:
        print(ii)

    xx = data[0].flatten()
    yy = data[1] == 1
    ggln1.predict(xx, target=[yy])
    ii+=1
    if ii>10000:
        break


correct=0
incorrect =0
ii = 0
for data in testset:

    y_out = ggln1.predict(data[0].flatten())
    # print(y_out)
    if np.round(y_out) == (data[1] == 1):
        correct += 1
        print(f'correct: {y_out}, {data[1]}')
    else:
        incorrect +=1
        print(f'incorrect: {y_out}, {data[1]}')


    ii +=1
    
    if ii%100==0:
        
        print(correct/(correct+incorrect))
