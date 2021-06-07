
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


trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)


testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False,transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

# make the GLNs (one for each MNIST class)
glns_set = []
num_classes =10
for ii in range(num_classes):
    glns_set.append(glns.GGLN([8,8,8,1], 28*28, 8, lr=3e-2, min_sigma_sq=0.05, bias_len=3,
                    init_bias_weights=[None, None, None]))

# train the GLNS
ii = 0
for data in trainset:
    if ii%100==0:
        print(ii)
    xx = data[0].flatten()
    # train each class separately
    for gln_i in range(num_classes):

        yy = data[1] == gln_i
        glns_set[gln_i].predict(xx, target=[yy])
    ii+=1
    if ii>1000: #break after however datapoints
        break



# Test the GLNs
correct=0
incorrect =0
ii = 0
predictions = np.zeros(num_classes)
for data in testset:

    # make predictions for each class
    for gln_i in range(num_classes):

        predictions[gln_i] = glns_set[gln_i].predict(data[0].flatten())

    # pick the class with the highest prediction
    if np.argmax(predictions) == data[1]:
        # print(f'correct: {np.argmax(predictions) }, {data[1]}')
        correct += 1
    else:
        # print(f'incorrect: {np.argmax(predictions) }, {data[1]}')

        incorrect +=1

    ii +=1
    
    if ii%100==0:
        
        print(correct/(correct+incorrect))
