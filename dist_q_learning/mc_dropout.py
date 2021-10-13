import torch as tc


class DNNModel(tc.nn.Module):

    def __init__(
            self, input_size, output_size, dropout_rate, hidden_sizes,
            sigmoid_vals=True):
        super(DNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        hidden_modules = [tc.nn.Linear(input_size, hidden_sizes[0])]
        for i in range(len(hidden_sizes)):
            hidden_modules.append(tc.nn.ReLU())
            if self.dropout_rate is not None:
                hidden_modules.append(tc.nn.Dropout(p=self.dropout_rate))
            if (i + 1) < len(hidden_sizes):
                next_size = hidden_sizes[i + 1]
            else:
                next_size = output_size
            hidden_modules.append(tc.nn.Linear(hidden_sizes[i], next_size))
        if sigmoid_vals:
            hidden_modules.append(tc.nn.Sigmoid())

        self.f = tc.nn.Sequential(*hidden_modules)

    def forward(self, x):
        return self.f(x)


class DropoutNet:

    def __init__(
            self,
            input_size,
            num_actions,
            num_horizons,
            feat_mean=0.5,
            lr=1e-2,
            name="Unnamed_mcd",
            n_samples=40,
            dropout_rate=0.5,
            hidden_sizes=(64, 32),
            sigmoid_vals=True,
            **kwargs
    ):
        self.samples = n_samples
        self.decay = 1e-6

        self.input_size = input_size
        self.num_actions = num_actions
        self.num_horizons = num_horizons
        self.output_size = num_actions * num_horizons
        self.feat_mean = feat_mean
        self.lr = lr
        self.lr_decay_per_step = 0.998
        self.min_lr = 0.0001
        self.name = name + "_MCD"

        self.net = DNNModel(
            self.input_size, self.output_size, dropout_rate=dropout_rate,
            hidden_sizes=hidden_sizes, sigmoid_vals=sigmoid_vals)
        print(f"{self.name}:")
        print(self.net)
        self.loss_f = tc.nn.MSELoss()
        self.optimizer = None
        self.lr_schedule = None
        self.make_optimizer()

    def make_optimizer(self):
        """Currently  implements a singleton, only producing schedule on
        2nd call (i.e. after burn in).

        Can be used to reset the optimizer if desired.
        """

        if self.optimizer is None:
            self.optimizer = tc.optim.Adam(self.net.parameters(), lr=self.lr)
        # if self.optimizer is None:
        #     self.optimizer = tc.optim.SGD(self.net.parameters(), lr=self.lr)
            # , weight_decay=self.decay)
        elif self.lr_schedule is None:
            self.lr_schedule = tc.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=300, gamma=0.5)

    def uncertainty_estimate(
            self, states, actions, quantile, horizon=None, debug=False):
        self.net.train()
        if horizon is None:
            horizon = self.num_horizons
        with tc.no_grad():
            y_samples = self.take_samples(states, actions, horizon)

        quantile_vals = tc.quantile(y_samples, quantile, dim=0)
        if debug:
            print(f"Uncertainty estimate for {self.name}, h{horizon}")
            print(f"Medians\n{tc.median(y_samples, dim=0)[0]}")
            print(f"Quantiles_{quantile}\n{quantile_vals}")

        return quantile_vals

    def take_samples(self, input_x, actions=None, horizon=None):
        """Return a sample-wise array of net predictions for x

        Operates for a specific time horizon on the net. Optionally
        indexed at actions.
        """
        assert horizon is not None
        out_size = [self.samples, input_x.shape[0]] + (
            [self.num_actions] if actions is None else [])
        y_samp = tc.zeros(*out_size)
        horizon_slice = slice(
            (horizon - 1) * self.num_actions, horizon * self.num_actions)
        for s in range(self.samples):
            sample_batch = self.net(input_x)
            sample_batch = sample_batch[..., horizon_slice]
            if actions is not None:
                sample_batch = tc.squeeze(
                    tc.gather(sample_batch, 1, actions), 1)
            y_samp[s] = sample_batch
        return y_samp

    def copy_values(self, new_weights):
        self.net.load_state_dict(new_weights)

    def update_learning_rate(self, lr):
        assert lr <= self.lr, (
            f"Not expecting changes to LR self {self.lr} -> {lr}")
        self.lr = lr

    def predict(self, inputs, actions=None, target=None, horizon=None):
        if horizon is not None and horizon <= 0:
            raise ValueError(horizon)
        elif horizon is None:
            horizon = self.num_horizons
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

        self.net.train()
        if target is None:
            assert actions is None
            with tc.no_grad():
                y_samp = self.take_samples(input_features, horizon=horizon)
            prediction_values = tc.mean(y_samp, dim=0)
            # prediction_values, _ = torch.median(y_samp, dim=0)
            return prediction_values
        else:
            y_pred = self.net(input_features)
            horizon_slice = slice(
                (horizon - 1) * self.num_actions, horizon * self.num_actions)
            y_pred = y_pred[..., horizon_slice]
            if y_pred.shape[-1] != 1:
                y_pred = tc.gather(y_pred, 1, actions)

            self.optimizer.zero_grad()
            loss = self.loss_f(y_pred, target)
            loss.backward()
            self.optimizer.step()
            if self.lr_schedule is not None\
                    and self.lr_schedule.get_last_lr()[0] > self.min_lr\
                    and (horizon is None or horizon == self.num_horizons):
                init_lr = self.lr_schedule.get_last_lr()
                self.lr_schedule.step()
                new_lr = self.lr_schedule.get_last_lr()
                if new_lr != init_lr:
                    print("Stepping learning rate", init_lr, "->", new_lr)
