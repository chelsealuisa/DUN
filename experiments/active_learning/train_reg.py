import os
import argparse
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from src.utils import Datafeed, DatafeedIndexed, cprint, mkdir
from src.datasets import load_flight, load_gap_UCI
from src.probability import depth_categorical_VI
from src.DUN.train_fc import train_fc_DUN
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_fc_models import arq_uncert_fc_resnet, arq_uncert_fc_MLP
from src.baselines.SGD import SGD_regression_homo
from src.baselines.dropout import dropout_regression_homo
from src.baselines.mfvi import MFVI_regression_homo
from src.baselines.training_wrappers import regression_baseline_net, regression_baseline_net_VI
from src.baselines.train_fc import train_fc_baseline
from src.acquisition_fns import acquire_samples

matplotlib.use('Agg')

tic = time()

uci_names = ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein']
uci_gap_names = ['boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                 'kin8nm_gap', 'naval_gap', 'protein_gap']

parser = argparse.ArgumentParser(description='Regression dataset running script')

parser.add_argument('--n_epochs', type=int, default=4000,
                    help='number of iterations performed by the optimizer (default: 4000)')
parser.add_argument('--dataset', help='which dataset to trian on',
                    choices=["flights"]+uci_names+uci_gap_names)
parser.add_argument('--split', help='dataset split to train on (default: 0)',
                    type=str, default="0")
parser.add_argument('--datadir', type=str, help='where to save dataset (default: ../data/)',
                    default='../data/')
parser.add_argument('--gpu', type=int, help='which GPU to run on (default: None)', default=None)
parser.add_argument('--inference', type=str, help='model to use (default: DUN)',
                    default='DUN', choices=['DUN', 'MFVI', 'Dropout', 'SGD'])
parser.add_argument('--num_workers', type=int, help='number of parallel workers for dataloading (default: 1)', default=1)
parser.add_argument('--N_layers', type=int, help='number of hidden layers to use (default: 2)', default=2)
parser.add_argument('--width', type=int, help='number of hidden units to use (default: 50)', default=50)
parser.add_argument('--batch_size', type=int, help='training chunk size (default: 128)', default=128)
parser.add_argument('--valprop', type=float, help='valprop that was used (default: 0.15)', default=0.15)
parser.add_argument('--savedir', type=str, help='where to save results (default: ./saves_regression/)',
                    default='./saves_regression/')
parser.add_argument('--overcount', type=int, help='how many times to count data towards ELBO (default: 1)', default=1)
parser.add_argument('--lr', type=float, help='learning rate (default: 1e-3)', default=1e-3)
parser.add_argument('--momentum', type=float, help='momentum (default: 0.9)', default=0.9)
parser.add_argument('--wd', type=float, help='weight_decay, (default: 0)', default=0)
parser.add_argument('--network', type=str,
                    help='model type when using DUNs (other methods use ResNet) (default: ResNet)',
                    default='ResNet', choices=['ResNet', 'MLP'])
parser.add_argument('--num', type=str, default="0",
                    help='training run (useful for ensembles and other repeated training) (default: 0)')
parser.add_argument('--n_queries', type=int, 
                    help='number of iterations for active learning (default: 20)', default=20)
parser.add_argument('--query_size', type=int, 
                    help='number of acquired data points in active learning (default: 10)',
                    default=10)

args = parser.parse_args()


# Some defaults
batch_size = args.batch_size
momentum = args.momentum
epochs = args.n_epochs
nb_its_dev = 5

name = '_'.join([args.inference, args.dataset, str(args.N_layers), str(args.width), str(args.lr), str(args.wd),
                 str(args.overcount)])
if args.network == 'MLP':
    name += '_MLP'

cuda = (args.gpu is not None)
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('cuda', cuda)


# Create datasets
if args.dataset == "flights":
    X_train, X_test, _, _, y_train, y_test, y_means, y_stds = load_flight(base_dir=args.datadir,
                                                                            k800=(args.split == "800k"))
elif args.dataset in uci_names + uci_gap_names:
    gap = False
    if args.dataset in uci_gap_names:
        gap = True
        args.dataset = args.dataset[:-4]

    X_train, X_test, _, _, y_train, y_test, y_means, y_stds = \
        load_gap_UCI(base_dir=args.datadir, dname=args.dataset, n_split=int(args.split), gap=gap)

trainset = DatafeedIndexed(torch.Tensor(X_train), torch.Tensor(y_train), transform=None)
testset = Datafeed(torch.Tensor(X_test), torch.Tensor(y_test), transform=None)

N_train = X_train.shape[0]
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

valloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=args.num_workers)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                        num_workers=args.num_workers)

# Experiment runs
n_runs = 5
results = np.zeros((args.n_queries, n_runs))
results_train = np.zeros((args.n_queries, n_runs))

for j in range(n_runs):
    seed = j
    mkdir(f'{args.savedir}/{name}/{j}')

    # Reset train data
    trainset.unlabeled_mask = np.ones(X_train.shape[0])

    # Instantiate model
    width = args.width
    n_layers = args.N_layers
    wd = args.wd
    lr = args.lr

    if args.inference == 'MFVI':
        prior_sig = 1

        model = MFVI_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers, prior_sig=1)

        net = regression_baseline_net_VI(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                        MC_samples=10, train_samples=5)

    elif args.inference == 'Dropout':
        model = dropout_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                        width=width, n_layers=n_layers, p_drop=0.1)

        net = regression_baseline_net(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                    MC_samples=10, weight_decay=wd)
    elif args.inference == 'SGD':
        model = SGD_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers)

        net = regression_baseline_net(model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=None,
                                    MC_samples=0, weight_decay=wd)
    elif args.inference == 'DUN':

        if args.network == 'ResNet':
            model = arq_uncert_fc_resnet(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
        elif args.network == 'MLP':
            model = arq_uncert_fc_MLP(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
        else:
            raise Exception('Bad network type. This should never raise as there is a previous assert.')

        prior_probs = [1 / (n_layers + 1)] * (n_layers + 1)
        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, N_train, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None,
                    regression=True, pred_sig=None, weight_decay=wd)


    # Active learning loop
    for i in range(args.n_queries):
        
        # Acquire data
        acquire_samples(net, trainset, args.query_size)
        n_labelled = int(sum(1 - trainset.unlabeled_mask))
        
        # Train model on labeled data
        labeled_idx = np.where(trainset.unlabeled_mask == 0)[0]
        labeledloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, num_workers=args.num_workers, sampler=torch.utils.data.SubsetRandomSampler(labeled_idx)
            )
        
        cprint('p', f'Query: {i}\tlabelled points: {len(labeled_idx)}')

        if args.inference in ['MFVI', 'Dropout', 'SGD']:
            marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
                approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
                train_fc_baseline(net, f'{name}/{j}', args.savedir, batch_size, epochs, labeledloader, valloader, cuda=cuda,
                            flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None, track_posterior=False, 
                            track_exact_ELBO=False, seed=seed, save_freq=nb_its_dev, basedir_prefix=n_labelled)
        else:
            marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
                approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
                train_fc_DUN(net, f'{name}/{j}', args.savedir, batch_size, epochs, labeledloader, valloader,
                        cuda, seed=seed, flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None,
                        track_posterior=True, track_exact_ELBO=False, tags=None,
                        load_path=None, save_freq=nb_its_dev, basedir_prefix=n_labelled)

        # Record performance
        min_train_error = min([err for err in err_train if err>0])
        min_val_error = min([err for err in err_dev if err>0])
        results_train[i,j] = min_train_error
        results[i,j] = min_val_error

        # Create plots
        media_dir = basedir + '/media'
        show_range = 5
        ylim = 3
        add_noise = False
        
        if args.inference == 'DUN':
            np.savetxt(f'{media_dir}/approx_d_posterior.csv', approx_d_posterior, delimiter=',')
            np.savetxt(f'{media_dir}/true_d_posterior.csv', true_d_posterior, delimiter=',')

            # Posterior over depth
            x = np.array([i for i in range(true_d_posterior.shape[1])])
            height_true = true_d_posterior[-1,:]
            height_approx = approx_d_posterior[-1,:]
            
            plt.figure(dpi=80)
            plt.bar(x, height_true)
            plt.title('Posterior distribution over depth')
            plt.xlabel('Layer')
            plt.savefig(f'{media_dir}/depth_post_true.png', format='png', bbox_inches='tight')
            plt.close()
            
            plt.figure(dpi=80)
            plt.bar(x, height_approx)
            plt.title('Approximate posterior distribution over depth')
            plt.xlabel('Layer')
            plt.savefig(f'{media_dir}/depth_post_approx.png', format='png', bbox_inches='tight')
            plt.close()

    cprint('p', f'Train errors: {results_train[:,j]}')
    cprint('p', f'Val errors: {results[:,j]}\n')
    np.savetxt(f'{args.savedir}/{name}/{j}/results_{j}.csv', results[:,j], delimiter=',')

    # plot validation error
    plt.figure(dpi=200)
    plt.plot(results[:,j])
    plt.xlabel('Query number')
    plt.ylabel('Validation error')
    plt.tight_layout()
    plt.savefig(f'{args.savedir}/{name}/{j}/val_error.png', format='png', bbox_inches='tight')

means = results.mean(axis=1).reshape(-1,1)
stds = results.std(axis=1).reshape(-1,1)
results = np.concatenate((means, stds, results), axis=1)
np.savetxt(f'{args.savedir}/{name}/results.csv', results, delimiter=',')

toc = time()
cprint('r', toc - tic)