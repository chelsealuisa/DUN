import random

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.utils import BaseNet, cprint, to_variable
from src.utils import rms
from src.probability import homo_Gauss_mloglike


def ensemble_predict(net, savefiles, x, return_model_std=False, return_individual_functions=False,  to_cpu=False):
    mean_vec = []
    noise_std_vec = []
    for file in savefiles:
        net.load(file, to_cpu=to_cpu)
        mu, noise_std = net.predict(x, return_model_std=False)  # want full std = noise std
        mean_vec.append(mu.data)
        noise_std_vec.append(noise_std.data)
    mean_vec = torch.stack(mean_vec, dim=0)
    std_vec = torch.stack(noise_std_vec, dim=0)

    if return_individual_functions:
        return mean_vec, std_vec
    else:
        mean = mean_vec.mean(dim=0)
        model_var = mean_vec.var(dim=0)
        if len(savefiles) == 0:
            model_var = torch.zeros_like(model_var)

        noise_var = std_vec.pow(2).mean(dim=0)

        if return_model_std:
            return mean, model_var.pow(0.5)
        else:
            pred_std = (model_var + noise_var).pow(0.5)
            return mean, pred_std


class regression_baseline_net(BaseNet):
    def __init__(self, model, N_train, lr=1e-2, momentum=0.5, cuda=True, schedule=None, seed=None,
                 weight_decay=0, MC_samples=10, regression=True, Adam=False):
        super(regression_baseline_net, self).__init__()

        cprint('y', 'DUN learnt with marginal likelihood categorical output')
        self.lr = lr
        self.momentum = momentum
        self.Adam = Adam
        self.weight_decay = weight_decay
        self.MC_samples = MC_samples
        self.model = model
        self.cuda = cuda
        self.seed = seed
        self.regression = regression
        if self.regression:
            self.f_neg_loglike = homo_Gauss_mloglike(self.model.output_dim, None)
            self.f_neg_loglike_test = self.f_neg_loglike
        else:
            self.f_neg_loglike = nn.CrossEntropyLoss(reduction='none')  # This one takes logits
            self.f_neg_loglike_test = nn.NLLLoss(reduction='none')  # This one takes log probs

        self.N_train = N_train
        self.create_net()
        self.create_opt()
        self.schedule = schedule  # [] #[50,200,400,600]
        if self.schedule is not None and len(self.schedule) > 0:
            self.make_scheduler(gamma=0.1, milestones=self.schedule)
        self.epoch = 0

    def create_net(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.cuda:
                torch.cuda.manual_seed(self.seed)
        if self.cuda:
            self.model.cuda()
            self.f_neg_loglike.cuda()

            cudnn.benchmark = True
        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        param_list = list(self.model.parameters()) + list(self.f_neg_loglike.parameters())
        if not self.Adam:
            self.optimizer = torch.optim.SGD(param_list, lr=self.lr, momentum=self.momentum,
                                             weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)

    def fit(self, x, y):
        """Standard training loop: MC dropout and Ensembles"""
        self.model.train()
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
            y = y.squeeze(1)
        self.optimizer.zero_grad()
        mean = self.model.forward(x)
        NLL = self.f_neg_loglike(mean, y).mean(dim=0)
        NLL.backward()
        self.optimizer.step()

        if self.regression:
            err = rms(mean, y).item()
        else:
            pred = mean.max(dim=1, keepdim=False)[1] # get the index of the max probability
            err = pred.ne(y.data).sum().item() / y.shape[0]
        
        return -NLL.data.item(), NLL.data.item(), err

    def fit_bias_reduction(self, x, y, idx, dataset):
        """Standard training loop (dropout and ensembles) with active learning bias reduction weights"""
        self.model.train()
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
            y = y.squeeze(1)
        self.optimizer.zero_grad()
        mean = self.model.forward(x)
        NLL_per_x = self.f_neg_loglike(mean, y)
        weights = dataset.get_rlure_weights(idx)
        weights = torch.reshape(weights, NLL_per_x.shape)
        NLL = NLL_per_x * weights
        NLL = NLL.mean(dim=0)
        NLL.backward()
        self.optimizer.step()

        if self.regression:
            err = rms(mean, y).item()
        else:
            pred = mean.max(dim=1, keepdim=False)[1] # get the index of the max probability
            err = pred.ne(y.data).sum().item() / y.shape[0]
        
        return -NLL.data.item(), NLL.data.item(), err

    def eval(self, x, y):
        self.model.eval()
        with torch.no_grad():
            x, y = to_variable(var=(x, y), cuda=self.cuda)
            if not self.regression:
                y = y.long()
            if self.regression:
                mean, model_std = self.model.forward_predict(x, self.MC_samples, softmax=(not self.regression))
                mean_pred_negloglike = self.f_neg_loglike(mean, y, model_std=model_std).mean(dim=0).data
                err = rms(mean, y).item()
            else:
                probs = self.model.forward_predict(x, self.MC_samples, softmax=(not self.regression))
                mean_pred_negloglike = self.f_neg_loglike_test(torch.log(probs), y).mean(dim=0).data
                pred = probs.max(dim=1, keepdim=False)[1]
                err = pred.ne(y.data).sum().item() / y.shape[0]
            
            return mean_pred_negloglike.item(), err

    def predict(self, x, Nsamples=10, return_model_std=False):
        self.model.eval()
        with torch.no_grad():
            x, = to_variable(var=(x,), cuda=self.cuda)

            if self.regression:
                mean, model_std = self.model.forward_predict(x, Nsamples, softmax=(not self.regression))
                if return_model_std:
                    return mean.data, model_std  # not data in order to take integer from sgd
                else:
                    pred_std = (model_std**2 + self.f_neg_loglike.log_std.exp()**2).pow(0.5)
                    return mean.data, pred_std.data
            else:
                probs = self.model.forward_predict(x, Nsamples, softmax=(not self.regression))
                return probs.data


class regression_baseline_net_VI(regression_baseline_net):
    def __init__(self,  model, N_train, lr=1e-2, momentum=0.5, cuda=True, schedule=None, seed=None,
                 weight_decay=0, MC_samples=10, train_samples=3, regression=True, Adam=False):
        super(regression_baseline_net_VI, self).__init__(model=model, N_train=N_train, lr=lr, momentum=momentum,
                                                         cuda=cuda, schedule=schedule, seed=seed, weight_decay=0,
                                                         MC_samples=MC_samples, regression=regression, Adam=Adam)
        # We fix weight decay to be 0 here as we are using KL divergence.
        self.train_samples = train_samples

    def fit(self, x, y):
        """Optimise stochastically estimated marginal joint of parameters and weights"""
        self.model.train()
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
            y = y.squeeze(1)
        self.optimizer.zero_grad()

        sample_means = self.model.forward(x, self.train_samples)
        batch_size = x.shape[0]
        repeat_dims = [self.train_samples] + [1 for i in range(1, len(y.shape))]  # Repeat batchwise without interleave
        y_expand = y.repeat(*repeat_dims)  # targets are same across acts -> interleave
        sample_means_flat = sample_means.view(self.train_samples * batch_size, -1)  # flattening results in batch_n changing first
        
        E_NLL = self.f_neg_loglike(sample_means_flat, y_expand).view(self.train_samples, batch_size).mean(dim=(0,1))
        minus_E_ELBO = E_NLL + self.model.get_KL() / self.N_train
        minus_E_ELBO.backward()
        self.optimizer.step()

        if self.regression:
            err = rms(sample_means.mean(dim=0), y).item()
        else:
            pred = sample_means.mean(dim=0).max(dim=1, keepdim=False)[1]
            err = pred.ne(y.data).sum().item() / y.shape[0]
        return -minus_E_ELBO.data.item(), E_NLL.data.item(), err

    def fit_bias_reduction(self, x, y, idx, dataset):
        """Optimise stochastically estimated marginal joint of parameters and weights"""
        self.model.train()
        x, y = to_variable(var=(x, y), cuda=self.cuda)
        if not self.regression:
            y = y.long()
            y = y.squeeze(1)
        self.optimizer.zero_grad()

        sample_means = self.model.forward(x, self.train_samples)
        batch_size = x.shape[0]
        repeat_dims = [self.train_samples] + [1 for i in range(1, len(y.shape))]  # Repeat batchwise without interleave
        y_expand = y.repeat(*repeat_dims)  # targets are same across acts -> interleave
        sample_means_flat = sample_means.view(self.train_samples * batch_size, -1)  # flattening results in batch_n changing first
        
        E_NLL_per_x = self.f_neg_loglike(sample_means_flat, y_expand).view(self.train_samples, batch_size).mean(dim=0)
        weights = dataset.get_rlure_weights(idx)
        weights = torch.reshape(weights, E_NLL_per_x.shape)
        E_NLL = E_NLL_per_x * weights
        E_NLL = E_NLL.mean(dim=0)
        minus_E_ELBO = E_NLL + self.model.get_KL() / self.N_train
        minus_E_ELBO.backward()
        self.optimizer.step()

        if self.regression:
            err = rms(sample_means.mean(dim=0), y).item()
        else:
            pred = sample_means.mean(dim=0).max(dim=1, keepdim=False)[1]
            err = pred.ne(y.data).sum().item() / y.shape[0]
        return -minus_E_ELBO.data.item(), E_NLL.data.item(), err
