import torch as tc


class DNNModel(tc.nn.Module):

    def __init__(self, input_size, output_size, dropout_rate):
        super(DNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.f = tc.nn.Sequential(
            tc.nn.Linear(input_size, 32),
            tc.nn.Tanh(),
            tc.nn.Dropout(p=self.dropout_rate),
            tc.nn.Linear(32, 64),
            tc.nn.Tanh(),
            tc.nn.Dropout(p=self.dropout_rate),
            tc.nn.Linear(64, output_size),
            tc.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.f(x)


class DropoutNet:

    def __init__(
            self,
            input_size,
            output_size,
            feat_mean=0.5,
            lr=1e-2,
            name="Unnamed_mcd",
            n_samples=40,
            dropout_rate=0.5,
            **kwargs
    ):
        self.samples = n_samples
        self.decay = 1e-6

        self.input_size = input_size
        self.output_size = output_size
        self.feat_mean = feat_mean
        self.lr = lr
        self.name = name + "_MCD"

        self.net = DNNModel(
            self.input_size, self.output_size, dropout_rate=dropout_rate)
        print(self.net)
        self.loss_f = tc.nn.MSELoss()
        self.optimizer = None
        self.make_optimizer()

    def make_optimizer(self):
        # self.optimizer = tc.optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = tc.optim.SGD(
            self.net.parameters(), lr=self.lr, weight_decay=self.decay)

    def uncertainty_estimate(self, states, actions, quantile, debug=False):
        self.net.train()
        with tc.no_grad():
            y_samples = self.take_samples(states, actions)

        quantile_vals = tc.quantile(y_samples, quantile, dim=0)
        if debug:
            print(f"Uncertainty estimate for {self.name}")
            print(f"Medians\n{tc.median(y_samples, dim=0)[0]}")
            print(f"Quantiles_{quantile}\n{quantile_vals}")

        return quantile_vals

    def take_samples(self, input_x, actions=None):
        """Return a sample-wise array of net predictions for x

        Optionally indexed at actions.
        """
        out_size = [self.samples, input_x.shape[0]] + (
            [self.output_size] if actions is None else [])
        y_samp = tc.zeros(*out_size)
        for s in range(self.samples):
            sample_batch = self.net(input_x)
            if actions is not None:
                sample_batch = tc.squeeze(
                    tc.gather(sample_batch, 1, actions), 1)
            y_samp[s] = sample_batch
        return y_samp

    def copy_values(self, new_weights):
        self.net.load_state_dict(new_weights)

    def update_learning_rate(self, lr):
        assert lr == self.lr, "Not expecting changes to LR"
        self.lr = lr

    def predict(self, inputs, actions=None, target=None):
        if not isinstance(inputs, tc.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(inputs)}")
        if target is not None and not isinstance(target, tc.Tensor):
            raise TypeError(f"Expected torch tensor, got {type(target)}")
        input_features = inputs.float()
        target = None if target is None else target.float()
        assert (target is None) == (actions is None)
        assert input_features.ndim == 2 and (
               target is None
               or target.shape == (input_features.shape[0], 1)), (
            f"Incorrect dimensions for input: {input_features.shape}"
            + ("" if target is None else f", or targets: {target.shape}"))

        if target is None:
            with tc.no_grad():
                y_samp = self.take_samples(input_features)
            prediction_values = tc.mean(y_samp, dim=0)
            # prediction_values, _ = torch.median(y_samp, dim=0)
            return prediction_values
        else:
            self.net.train()
            y_pred = self.net(input_features)
            if y_pred.shape[1] != 1:
                y_pred = tc.gather(y_pred, 1, actions)

            self.optimizer.zero_grad()
            loss = self.loss_f(y_pred, target)
            loss.backward()
            self.optimizer.step()
