
from numpy.core.fromnumeric import argmax
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
num_classes =10
for ii in range(num_classes):
    glns_set.append(glns.GGLN([8,8,8,1], 28*28, 8, lr=3e-2, min_sigma_sq=0.05, bias_len=3,
                    init_bias_weights=[None, None, None]))

ii = 0
for data in trainset:
    if ii%100==0:
        print(ii)
        # ggln1.set_bais_weights([1, 4, 1])

    xx = data[0].flatten()
    for gln_i in range(num_classes):

        yy = data[1] == gln_i
        glns_set[gln_i].predict(xx, target=[yy])
    ii+=1
    if ii>1000:
        break

# ii = 0
# for data in trainloader:
#     if ii%100==0:
#         print(ii)
#         # ggln1.set_bais_weights([1, 4, 1])

#     xx = data[0].flatten()
#     for gln_i in range(10):

#         yy = data[1] == gln_i
#         glns_set[gln_i].predict(xx, target=yy)
#     ii+=1
#     if ii>1000:
#         break



correct=0
incorrect =0
ii = 0
predictions = np.zeros(num_classes)
for data in testset:

    for gln_i in range(num_classes):

        # yy = data[1] == gln_i
        predictions[gln_i] = glns_set[gln_i].predict(data[0].flatten())

    if np.argmax(predictions) == data[1]:
        print(f'correct: {np.argmax(predictions) }, {data[1]}')
        correct += 1
    else:
        print(f'incorrect: {np.argmax(predictions) }, {data[1]}')

        incorrect +=1

    ii +=1
    
    if ii%100==0:
        
        print(correct/(correct+incorrect))
