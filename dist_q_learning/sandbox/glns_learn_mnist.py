import jax
import torch as tc
import numpy as np
import torchvision
import torchvision.transforms as transforms

import glns


def set_gpu():
    cuda_available = tc.cuda.is_available()
    print("Torch GPU available?", cuda_available)
    print("Jax GPU available?", jax.devices())
    if not cuda_available:
        return False
    # Set the pytorch GPU as necessary
    print("Current device", tc.cuda.current_device())
    n_device = tc.cuda.device_count()
    print("Avaialable devices", n_device)
    for i in range(n_device):
        print(f"Device {i}: {tc.cuda.get_device_name(i)}")
    dev_i = int(input("Input device number (or 'cpu'): "))
    tc.cuda.device(dev_i)
    return True


def get_data(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (1,))])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    trainloader = tc.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    testloader = tc.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def get_models(num_classes, batch_size):
    """Make num_classes GLNs - one for each output"""
    glns_set = [
        glns.GGLN(
            layer_sizes=[8, 8, 8, 1],
            input_size=28 * 28,
            context_dim=8,
            batch_size=batch_size,
            lr=3e-2,
            min_sigma_sq=0.05,
            bias_len=3,
            init_bias_weights=[None, None, None])
        for _ in range(num_classes)]
    return glns_set


def train(glns, trainset):
    # train the GLNS
    data_len = len(trainset)
    max_steps = None
    if max_steps is not None:
        assert max_steps <= data_len
    print(f"Training on {max_steps or data_len} batches...")
    for batch_i, (x, y) in enumerate(trainset):
        flat_x = x.flatten(start_dim=1)
        if batch_i % (data_len // 20) == 0:
            print(f"{100 * batch_i // data_len}% ({batch_i}/{data_len})")
        if batch_i == 0:
            print("Batch representative")
            print("data", x.shape)
            print("Prepped data", flat_x.shape)
            print("Label", y.shape)
        # train each class separately
        for class_i, gln_classifier in enumerate(glns):
            gln_y = y == class_i
            gln_classifier.predict(flat_x, target=gln_y)

        if max_steps is not None and batch_i > max_steps:
            break


def test(glns, testset):
    """

    TODO:
        Get batch size from dataset rather than GLNs as it's
        training only
    """
    data_len = len(testset)
    # Test the GLNs
    num_classes = len(glns)
    correct = np.zeros(num_classes)
    incorrect = np.zeros(num_classes)
    total = np.zeros(num_classes)
    print(f"Testing on {glns[0].batch_size * data_len} images...")
    for batch_i, (x, y) in enumerate(testset):
        batch_scores = np.zeros((glns[0].batch_size, num_classes))
        flat_x = x.flatten(start_dim=1)
        # make predictions for each class
        for i, gln_classifier in enumerate(glns):
            batch_scores[:, i] = gln_classifier.predict(flat_x)

        predictions = np.argmax(batch_scores, axis=-1)
        # Pick the class with the highest prediction
        for pred, label in zip(predictions, y.numpy()):
            incorrect[label] += (pred != label).astype(int)
            correct[label] += (pred == label).astype(int)
            total[label] += 1

    print("Per (true) class evaluation:")
    for c in range(num_classes):
        print(f"{c}: {100 * correct[c] / total[c]:.2f}% "
              f"({correct[c]}, {incorrect[c]} / {total[c]})")

    print("Total accuracy:")
    print(f"{100 * correct.sum() / total.sum():.2f}%")


if __name__ == "__main__":
    _batch_size = 16
    set_gpu()
    train_data, test_data = get_data(batch_size=_batch_size)
    gln_list = get_models(
        num_classes=len(train_data.dataset.classes),
        batch_size=_batch_size)
    train(gln_list, train_data)
    test(gln_list, test_data)
