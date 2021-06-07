import torch as tc
import numpy as np
import torchvision
import torchvision.transforms as transforms

import glns


def get_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    trainloader = tc.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = tc.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_models(num_classes):
    """Make num_classes GLNs - one for each output"""
    glns_set = [
        glns.GGLN(
            [8, 8, 8, 1], 28 * 28, 8, lr=3e-2, min_sigma_sq=0.05,
            bias_len=3, init_bias_weights=[None, None, None])
        for _ in range(num_classes)]
    return glns_set


def train(glns, trainset):
    # train the GLNS
    ii = 0
    for data in trainset:
        if ii % 100 == 0:
            print(ii)
        xx = data[0].flatten()
        # train each class separately
        for gln_i in range(len(glns)):
            yy = data[1] == gln_i
            glns[gln_i].predict(xx, target=[yy])
        ii += 1
        # break after however datapoints
        if ii > 1000:
            break


def test(glns, testset):
    # Test the GLNs
    num_classes = len(glns)
    correct = 0
    incorrect = 0
    ii = 0
    predictions = np.zeros(num_classes)
    for data in testset:
        # make predictions for each class
        for gln_i in range(num_classes):
            predictions[gln_i] = glns[gln_i].predict(data[0].flatten())

        # pick the class with the highest prediction
        if np.argmax(predictions) == data[1]:
            # print(f'correct: {np.argmax(predictions) }, {data[1]}')
            correct += 1
        else:
            # print(f'incorrect: {np.argmax(predictions) }, {data[1]}')
            incorrect += 1

        ii += 1
        if ii % 100 == 0:
            print(correct/(correct+incorrect))


if __name__ == "__main__":
    train_data, test_data = get_data()
    glns = get_models(len(train_data.dataset.classes))
    train(glns, train_data)
    test(glns, test_data)
