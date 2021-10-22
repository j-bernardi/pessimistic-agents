import torch as tc
import numpy as np

from mc_dropout import DNNModel, DropoutNet


def f(x):
    MIN = 0.7
    return MIN + (
        (1. - MIN) * (np.exp(0.5) / 0.5) * np.abs(x / np.exp(2 * (x ** 2))))


if __name__ == "__main__":
    N = 10000
    SPLIT = 8000
    B = 32
    SIZE = 1
    EPOCH = 20

    x_data = tc.distributions.Uniform(low=-1, high=1).sample((N, SIZE))
    y_data = f(x_data)

    x_train, y_train = x_data[:SPLIT], y_data[:SPLIT]
    x_test, y_test = x_data[SPLIT:], y_data[SPLIT:]

    model = DNNModel(
        SIZE, 1, dropout_rate=0., hidden_sizes=(256, 256, 256), sigmoid_vals=False)

    loss_f = tc.nn.SmoothL1Loss()
    optimizer = tc.optim.Adam(model.parameters(), lr=0.01)
    stepper = tc.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    last_loss = None
    counter = 0

    for epoch in range(20):
        for i in range(0, x_train.shape[0], B):
            samples = tc.randint(0, x_train.shape[0], size=(B,))
            x = x_train[samples]
            y_gt = y_train[samples]

            y_pred = model(x)
            assert y_pred.shape == y_gt.shape
            loss = loss_f(y_pred, y_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training loss: {loss:.5f}")

        test_pred = model(x_test)
        test_loss = loss_f(test_pred, y_test)
        print(f"EPOCH loss on test {test_loss}")
        if last_loss is not None:
            if test_loss > last_loss:
                counter += 1
            else:
                last_loss = test_loss
            if counter == 10:
                print(f"STEP LR to")
                stepper.step()
                counter = 0
                last_loss = test_loss
        else:
            last_loss = test_loss

    print(f"FINAL loss {test_loss}")
    print(f"GT\n{y_test[:20].squeeze()}")
    print(f"PRED\n{test_pred[:20].squeeze()}")

    print(model)
    for n, p in model.named_parameters():
        print("PARAM", n, p.shape)
