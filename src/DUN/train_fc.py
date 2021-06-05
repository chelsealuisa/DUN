import os
import time
import tempfile

import numpy as np
import torch
import torch.utils.data

from src.utils import mkdir, cprint


def train_fc_DUN(net, name, save_dir, batch_size, nb_epochs, train_loader, val_loader,
              cuda, seed, flat_ims=False, nb_its_dev=1, early_stop=None,
              track_posterior=False, track_exact_ELBO=False, tags=None,
              load_path=None, save_freq=None, q_nograd_its=0, basedir_prefix=None):

    rand_name = next(tempfile._get_candidate_names())
    if basedir_prefix:
        basedir = os.path.join(save_dir, name, f'{basedir_prefix}_{rand_name}')
        trainsize = int(basedir_prefix)
        x, y, *_ = next(iter(train_loader))
        if x.shape[1]==1:
            per_example_errors = np.zeros((3*nb_epochs, trainsize+1))
        else:
            per_example_errors = np.zeros((2*nb_epochs, trainsize+1))
    else:
        basedir = os.path.join(save_dir, name, rand_name)

    media_dir = basedir + '/media/'
    models_dir = basedir + '/models/'
    mkdir(models_dir)
    mkdir(media_dir)

    if seed is not None:
        torch.manual_seed(seed)

    if cuda and seed is not None:
        torch.cuda.manual_seed(seed)

    epoch = 0

    # train
    marginal_loglike_estimate = np.zeros(nb_epochs)
    # we can use this ^ to approximately track the true value by averaging batches
    train_mean_predictive_loglike = np.zeros(nb_epochs)
    dev_mean_predictive_loglike = np.zeros(nb_epochs)
    err_train = np.zeros(nb_epochs)
    err_dev = np.zeros(nb_epochs)

    true_d_posterior = []
    approx_d_posterior = []
    true_likelihood = []
    exact_ELBO = []

    best_epoch = 0
    best_marginal_loglike = -np.inf
    # best_dev_err = -np.inf
    # best_dev_ll = -np.inf

    if q_nograd_its > 0:
        net.prob_model.q_logits.requires_grad = False
    
    tic0 = time.time()
    for i in range(epoch, nb_epochs):
        if q_nograd_its > 0 and i == q_nograd_its:
            net.prob_model.q_logits.requires_grad = True

        net.set_mode_train(True)
        tic = time.time()
        nb_samples = 0
        xs = []
        ys = []
        errs = []
        for x, y, *_ in train_loader:
            if flat_ims:
                x = x.view(x.shape[0], -1)

            marg_loglike_estimate, minus_loglike, err, example_errors = net.fit(x, y)

            marginal_loglike_estimate[i] += marg_loglike_estimate * x.shape[0]
            err_train[i] += err * x.shape[0]
            train_mean_predictive_loglike[i] += minus_loglike * x.shape[0]
            nb_samples += len(x)

            if x.shape[1]==1:
                xs.extend(x.cpu().detach().numpy().reshape(-1))
            ys.extend(y.cpu().detach().numpy().reshape(-1))
            errs.extend(example_errors.cpu().detach().numpy().reshape(-1))

        marginal_loglike_estimate[i] /= nb_samples
        train_mean_predictive_loglike[i] /= nb_samples
        err_train[i] /= nb_samples

        toc = time.time()
        
        if x.shape[1]==1:
            per_example_errors[i*3:(i*3+3),0] = str(i)
            per_example_errors[i*3,1:] = np.array(xs)
            per_example_errors[i*3+1,1:] = np.array(ys)
            per_example_errors[i*3+2,1:] = np.array(errs)
        else:
            per_example_errors[i*2:(i*2+2),0] = str(i)
            per_example_errors[i*2,1:] = np.array(ys)
            per_example_errors[i*2+1,1:] = np.array(errs)
                
        # TODO: make work for both toy and reg datasets
        #per_example_errors[i*2:(i*2+2),0] = str(i)
        #per_example_errors[i*2,1:] = y.cpu().detach().numpy().reshape(-1)
        #per_example_errors[i*2+1,1:] = example_errors.cpu().detach().numpy().reshape(-1)
        #per_example_errors[i*3:(i*3+3),0] = str(i)
        #per_example_errors[i*3,1:] = x.cpu().detach().numpy().reshape(-1)
        #per_example_errors[i*3+1,1:] = y.cpu().detach().numpy().reshape(-1)
        #per_example_errors[i*3+2,1:] = example_errors.cpu().detach().numpy().reshape(-1)

        # ---- print
        print('\n depth approx posterior', net.prob_model.current_posterior.data.cpu().numpy())
        print("it %d/%d, ELBO/evidence %.4f, pred minus loglike = %f, err = %f" %
              (i, nb_epochs, marginal_loglike_estimate[i], train_mean_predictive_loglike[i], err_train[i]), end="")
        print(f'\n max error: {max(abs(example_errors)).item()}')

        cprint('r', '   time: %f seconds\n' % (toc - tic))
        net.update_lr()

        if track_posterior:
            approx_d_posterior.append(net.prob_model.current_posterior.data.cpu().numpy())
            exact_posterior, log_marginal_over_depth = net.get_exact_d_posterior(train_loader, train_bn=True,
                                                                                 logposterior=False)
            true_d_posterior.append(exact_posterior.data.cpu().numpy())
            true_likelihood.append(log_marginal_over_depth)

        if track_exact_ELBO:
            exact_ELBO.append(net.get_exact_ELBO(train_loader, train_bn=True))

        # ---- dev
        if i % nb_its_dev == 0:
            tic = time.time()
            nb_samples = 0
            for x, y in val_loader:
                if flat_ims:
                    x = x.view(x.shape[0], -1)

                minus_loglike, err = net.eval(x, y)

                dev_mean_predictive_loglike[i] += minus_loglike * x.shape[0]
                err_dev[i] += err * x.shape[0]
                nb_samples += len(x)

            dev_mean_predictive_loglike[i] /= nb_samples
            err_dev[i] /= nb_samples
            toc = time.time()

            cprint('g', '     pred minus loglike = %f, err = %f\n' % (dev_mean_predictive_loglike[i], err_dev[i]), end="")
            cprint('g', '    time: %f seconds\n' % (toc - tic))

        if save_freq is not None and i % save_freq == 0:
            net.save(models_dir + '/theta_last.dat')

        if marginal_loglike_estimate[i] > best_marginal_loglike:
            best_marginal_loglike = marginal_loglike_estimate[i]

            # best_dev_ll = dev_mean_predictive_loglike[i]
            # best_dev_err = err_dev[i]
            best_epoch = i
            cprint('b', 'best marginal loglike: %f' % best_marginal_loglike)
            if i % 2 == 0:
                net.save(models_dir + '/theta_best.dat')

        if early_stop is not None and (i - best_epoch) > early_stop:
            cprint('r', '   stopped early!\n')
            break

    toc0 = time.time()
    runtime_per_it = (toc0 - tic0) / float(i + 1)
    cprint('r', '   average time: %f seconds\n' % runtime_per_it)

    # fig cost vs its
    if track_posterior:
        approx_d_posterior = np.stack(approx_d_posterior, axis=0)
        true_d_posterior = np.stack(true_d_posterior, axis=0)
        true_likelihood = np.stack(true_likelihood, axis=0)
    if track_exact_ELBO:
        exact_ELBO = np.stack(exact_ELBO, axis=0)

    np.savetxt(f'{media_dir}/per_example_errors.csv', per_example_errors, delimiter=',')

    return marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
           approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir
