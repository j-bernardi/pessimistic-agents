import numpy as np

import torch


class GGLN():


    def __init__(self,
                 layer_sizes,
                 input_size,
                 context_map_size,
                 bias_len=3,
                 context_bias=True,
                 learning_rate=1e-4,
                 weight_clipping=5,
                 input_sig_sq=1.,
                 min_sig_sq=0.5,
                 max_sig_sq=1e3,
                 min_mu = 1e5,
                 max_mu = -1e5):


        torch.autograd.set_detect_anomaly(True)

        self.weights = []
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.input_size = input_size
        self.bias_len = bias_len
        self.context_map_size = context_map_size
        self.context_bias = context_bias
        self.learning_rate = learning_rate
        self.weight_clipping = weight_clipping
        self.input_sig_sq = input_sig_sq

        self.min_weight = -weight_clipping *0 + 1e-3*1
        self.max_weight = weight_clipping
        self.min_sig_sq = min_sig_sq
        self.max_sig_sq = max_sig_sq
        self.min_mu = min_mu
        self.max_mu = max_mu

        self.context_maps = []
        self.context_maps_bias =[]

        self.barrier_constant = 10

        self.boolean_converter = torch.tensor([[2 ** i for i in range(self.context_map_size)[::-1]]])

        self.weights.append(
            torch.full((2 ** self.context_map_size, self.input_size + self.bias_len, self.layer_sizes[0]),1. / ( self.input_size + self.bias_len), requires_grad=True))

        for ii in range(len(layer_sizes)-1):
            layer_shape=(2 ** self.context_map_size, self.layer_sizes[ii] + self.bias_len, self.layer_sizes[ii+1])
            self.weights.append(torch.full(layer_shape, 1. / (self.layer_sizes[ii] + self.bias_len), requires_grad=True))
            self.context_maps.append(torch.randn(size=(self.context_map_size, input_size, self.layer_sizes[ii] + self.bias_len)))
            if self.context_bias:
                self.context_maps_bias.append(torch.randn(size=(self.context_map_size, 1)))
            else:
                self.context_maps_bias.append(torch.zeros((self.context_map_size, 1)))

        self.context_maps.append(torch.randn(size=(self.context_map_size, input_size, self.layer_sizes[-1] + self.bias_len)))
        if self.context_bias:
            self.context_maps_bias.append(torch.randn(size=(self.context_map_size, 1)))
        else:
            self.context_maps_bias.append(torch.zeros((self.context_map_size, 1)))

        self.bias_mu = torch.linspace(-self.weight_clipping, self.weight_clipping, bias_len).reshape((1, bias_len))

        self.optim = torch.optim.SGD(self.weights, lr=self.learning_rate)

        print(self.weights)

    def context_function(self, inputs, layer_i, neuron_k):

        binary_region_indices = torch.tensor((torch.matmul(self.context_maps[layer_i][:, :, neuron_k], inputs.T) - self.context_maps_bias[layer_i]) > 0, dtype=torch.long)#.astype(torch.int)
        # binary_region_indices = (torch.matmul(self.context_maps[layer_i][:, :, neuron_k], inputs.T) - self.context_maps_bias[layer_i]) > 0

        weight_indices = torch.matmul(self.boolean_converter, binary_region_indices)
        return weight_indices


    def predict(self, input, side_info=None, target=None):
        # print(self.weights)

        input = torch.FloatTensor(input)

        if side_info is None:
            side_info = input[:]
        
        batch_size = input.shape[0]
        input = torch.cat((input, self.bias_mu.repeat(batch_size,1)),1)
        sig_sq = torch.full(input.shape, self.input_sig_sq)
        # print(input)
        mu = input[:]

        # print(self.layer_sizes)
        # print([w.shape for w in self.weights])

        for ii in range(self.num_layers):
            # print(f'layer: {ii}')
            sig_sq_new_save = torch.ones((batch_size, self.layer_sizes[ii]))
            mu_new_save = torch.ones((batch_size, self.layer_sizes[ii]))
            for k in range(self.layer_sizes[ii]):
                # print(f'neuron: {k}')

                weight_indices = self.context_function(side_info, ii, k)
                # w = torch.tensor(self.weights[ii][weight_indices, :, k], requires_grad=True)
                w = self.weights[ii][weight_indices, :, k]
                # w = w.squeeze(0)

                w = self.project_weights(w, sig_sq)



                # w = torch.clamp(w, self.min_weight, self.max_weight)

                sig_sq_new = 1. / torch.sum(w / sig_sq, dim=-1)


                mu_new = sig_sq_new * torch.sum(w * mu / sig_sq, dim=-1)
                
                # sig_sq_new = torch.clamp(sig_sq_new, self.min_sig_sq, self.max_sig_sq)
                # mu_new = torch.clamp(mu_new, self.min_mu, self.max_mu)

                # print(mu_new.shape)
                if target is not None:
                    # loss = torch.tensor(torch.log(sig_sq_new) + (target - mu_new)**2/sig_sq_new, requires_grad=True)
                    # loss= torch.sum(torch.log(sig_sq_new) + (target - mu_new)**2/sig_sq_new)
                    losses = torch.log(sig_sq_new) + (target.T - mu_new)**2/sig_sq_new

                    # losses = losses + self.barrier_function(w, sig_sq_new, mu_new)
                    # print(losses.shape)
                    # w.retain_grad()
                    # loss.backward()
                    for b in range(batch_size):
                        loss = losses.T[b]
                        weight_index = weight_indices.T[b]
                        # print(loss.shape)
                        torch.autograd.backward(loss, retain_graph=True )
                        # print(torch.autograd.grad(loss, w, retain_graph=True,allow_unused=True))
                        # print(self.weights[ii].grad[weight_index, :, k])
                        # print(self.weights[0].shape)
                        dweights= self.weights[ii].grad
                        dw = dweights[weight_index, :,k]
                        with torch.no_grad():
                            # print(self.weights[ii][weight_index, :, k].shape)
                            # print(w.shape)
                            # self.weights[ii][weight_index, :, k] = w[0,b,:]
                            self.weights[ii][weight_index, :, k] = self.weights[ii][weight_index, :, k] - self.learning_rate * dw
                            self.weights[ii].clamp(self.min_weight, self.max_weight)

                        # self.optim.step()
                    # self.optim.zero_grad()

                        # w = torch.clamp(w, self.min_weight, self.max_weight)

            
                # print(sig_sq_new.shape)
                sig_sq_new_save[:, k] = sig_sq_new
                mu_new_save[:, k] = mu_new
                sig_sq_new.clamp(self.min_sig_sq, self.max_sig_sq)
                mu_new.clamp(self.min_mu, self.max_mu)
                self.weights[ii].clamp(self.min_weight, self.max_weight)

            mu = torch.cat((mu_new_save, self.bias_mu.repeat(batch_size,1)),1)
            sig_sq = torch.cat((sig_sq_new_save, torch.ones((batch_size, self.bias_len))),1)
            sig_sq.clamp(self.min_sig_sq, self.max_sig_sq)
            mu.clamp(self.min_mu, self.max_mu)
            self.weights[ii].clamp(self.min_weight, self.max_weight)

        # print(self.weights)
        return mu_new, sig_sq_new

    def project_weights(self, w, sigma_sq_in):

        sigma_sq_in = sigma_sq_in

        w = torch.clamp(w, self.min_weight, self.max_weight)

        lambda_in = 1. / sigma_sq_in
        

        # sigma_sq_out = 1. / w.dot(lambda_in)

        sigma_sq_out = 1. / torch.sum(w * lambda_in, dim=1)
        sigma_sq_out = sigma_sq_out.unsqueeze(1)
        # If w.dot(x) < U, linearly project w such that w.dot(x) = U.

        # if any(sigma_sq_out < self.min_sig_sq):
        #     print('sigma_sq_out < self.min_sig_sq')
        #     print(w)

        w = torch.where(
            sigma_sq_out < self.min_sig_sq, w - lambda_in *
            (1. / sigma_sq_out - 1. / self.min_sig_sq) / torch.sum(lambda_in**2),
            w)

        # If w.dot(x) > U, linearly project w such that w.dot(x) = U.
        v1 = lambda_in * (1. / sigma_sq_out - 1. / self.max_sig_sq) / torch.sum(lambda_in**2)

        w = torch.where(
            sigma_sq_out > self.max_sig_sq, w - v1,
            w)
            
        # if any(sigma_sq_out > self.max_sig_sq):
        #     print('sigma_sq_out > self.max_sig_sq')
        #     print(w)
        # w = w.unsqueeze(0)
        # print(w.shape)
        return w



    # def project_weights(self, w, sig_sq):

    #     # this is stolen from deepmind's implementation
    #     # I think it is bad and wrong
    #     # but I will use it for the mean time

    #     # w = torch.minimum(torch.maximum(w, torch.FloatTensor([self.min_weight])), torch.FloatTensor([self.max_weight]))

    #     w = torch.clamp(w, self.min_weight, self.max_weight)

    #     sig_sq_new = 1 / torch.sum(w / sig_sq)

    #     w = torch.where(sig_sq_new < self.min_sig_sq,
    #             w - 1. / sig_sq * (1. / sig_sq_new - 1. / self.min_sig_sq)/torch.sum(1. / sig_sq),
    #             w)


    #     w = torch.where(sig_sq_new > self.max_sig_sq,
    #         w - 1. / sig_sq * (1. / sig_sq_new - 1. / self.max_sig_sq)/torch.sum(1. / sig_sq),
    #         w)

    #     return w

    def barrier_function(self, w, sig_sq, mu):

        phi = torch.sum(-1. * torch.log(w - self.min_weight) - torch.log(self.max_weight - w), dim=-1)

        phi = phi - torch.log(sig_sq - self.min_sig_sq) - torch.log(self.max_sig_sq - sig_sq)

        phi = phi - torch.log(mu - self.min_mu) - torch.log(self.max_mu - mu)

        return phi * self.barrier_constant


        




    # def predict(self, input, side_info=None, target=None):
    #     # print(self.weights)

    #     input = torch.FloatTensor(input)

    #     if side_info is None:
    #         side_info = input[:]

    #     input = torch.cat((input, self.bias_mu),1)
    #     sig_sq = torch.full(input.shape, self.input_sig_sq)
    #     # print(input)
    #     mu = input[:]

    #     # print(self.layer_sizes)
    #     # print([w.shape for w in self.weights])

    #     for ii in range(self.num_layers):
    #         # print(f'layer: {ii}')
    #         sig_sq_new_save = torch.ones((self.layer_sizes[ii]))
    #         mu_new_save = torch.ones((self.layer_sizes[ii]))
    #         for k in range(self.layer_sizes[ii]):
    #             # print(f'neuron: {k}')

    #             weight_indices = self.context_function(side_info, ii, k)
    #             # w = torch.tensor(self.weights[ii][weight_indices, :, k], requires_grad=True)
    #             w = self.weights[ii][weight_indices, :, k]
                
    #             w_proj = self.project_weights(w, sig_sq)

    #             sig_sq_new = 1. / torch.sum(w_proj / sig_sq)

    #             mu_new = sig_sq_new * torch.sum(w_proj * mu / sig_sq)

    #             if target is not None:
    #                 # loss = torch.tensor(torch.log(sig_sq_new) + (target - mu_new)**2/sig_sq_new, requires_grad=True)
    #                 loss = torch.log(sig_sq_new) + (target - mu_new)**2/sig_sq_new
    #                 # w.retain_grad()
    #                 # loss.backward()
                    
    #                 torch.autograd.backward(loss, retain_graph=True )
    #                 # print(torch.autograd.grad(loss, w, retain_graph=True,allow_unused=True))
    #                 # print(w.grad)
    #                 # print(self.weights[0].shape)

    #                 self.optim.step()
    #                 self.optim.zero_grad()
            
    #             sig_sq_new_save[k] = sig_sq_new
    #             mu_new_save[k] = mu_new

    #         mu = torch.cat((mu_new_save.unsqueeze(0), self.bias_mu),1)
    #         sig_sq = torch.cat((sig_sq_new_save, torch.ones((self.bias_len))))
        
    #     # print(self.weights)
    #     return mu_new, sig_sq_new