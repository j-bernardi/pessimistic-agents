import torch as tc
from mc_dropout import DNNModel
import copy
import torch.autograd.functional

class HessianNet:


    def __init__(
            self,
            input_size,
            num_actions,
            num_horizons,
            feat_mean=0.5,
            lr=1e-2,
            name="Unnamed_hessnet",
            dropout_rate=None,
            hidden_sizes=(64, 32),
            sigmoid_vals=True,
            **kwargs
    ):
        self.decay = 1e-6

        self.input_size = input_size
        self.num_actions = num_actions
        self.num_horizons = num_horizons
        self.output_size = num_actions * num_horizons
        self.feat_mean = feat_mean
        self.lr = lr
        self.lr_decay_per_step = 0.998
        self.min_lr = 0.0001
        self.name = name + "_HES"

        self.net = DNNModel(
            self.input_size, self.output_size, dropout_rate=dropout_rate,
            hidden_sizes=hidden_sizes, sigmoid_vals=sigmoid_vals)
        print(f"{self.name}:")
        print(self.net)
        self.loss_f = tc.nn.MSELoss()
        self.loss_f_no_reduce = tc.nn.MSELoss(reduction='none')
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


    def predict(self, inputs, actions=None, target=None):

        if target is None:
            self.net.eval()
            predictions = self.net(inputs)
            return predictions

        else:
            
            self.net.train()
            predictions = self.net(inputs)

            if predictions.shape[-1]  != 1:
                predictions = tc.gather(predictions, 1, actions)

            self.optimizer.zero_grad()

            loss = self.loss_f(predictions, target)
            loss.backward()

            self.optimizer.step()


    

    def uncertainty_estimate(
            self, states, x_whole, y_whole, actions=None
        ):

        init_weights = copy.deepcopy(self.net.state_dict)

        initial_estimates = self.predict(states)
        self.net.train()



        pred = self.net(x_whole)

        # loss = self.loss_f(pred, y_whole[ii])
        
        hess = torch.autograd.functional.hessian(
            self.loss_f_no_reduce, (pred, y_whole), create_graph=True)

        hess_sum = tc.sum(hess, dim=0)
        hess_inv = tc.inverse(hess_sum)

        fake_targets = [0., 1.]
        
        # for s in states:
        for fake_target in fake_targets:
            
            preds = self.net(states)

            if actions is not None:
                preds = tc.gather(preds, 1, actions)
                
            jac = torch.autograd.functional.jacobian(
                self.loss_f_no_reduce, (preds, tc.full((states.shape[0],), fake_target)),
                create_graph=True)
            
            for jac_single in jac:

                delta_w = tc.dot(hess_inv, jac_single)

                self.net.load_state_dict(init_weights + delta_w)




                
                
