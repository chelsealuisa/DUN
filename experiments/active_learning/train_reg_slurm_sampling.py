import os
import argparse
from time import time
import sys
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pl
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
from src.plots import plot_al_results, plot_mean_d_posterior

matplotlib.use('Agg')

tic = time()

uci_names = ['boston', 'concrete', 'energy', 'power', 'wine', 'yacht', 'kin8nm', 'naval', 'protein']
uci_gap_names = ['boston_gap', 'concrete_gap', 'energy_gap', 'power_gap', 'wine_gap', 'yacht_gap',
                 'kin8nm_gap', 'naval_gap', 'protein_gap']

parser = argparse.ArgumentParser(description='Regression dataset running script')

parser.add_argument('--n_epochs', type=int, default=4000,
                    help='number of iterations performed by the optimizer (default: 4000)')
parser.add_argument('--dataset', help='which dataset to trian on',
                    choices=['flights']+uci_names+uci_gap_names)
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
parser.add_argument('--output_folder', type=str, help='where to save results (default: ./saves_regression/)',
                    default='./saves_regression/')
parser.add_argument('--overcount', type=int, help='how many times to count data towards ELBO (default: 1)', default=1)
parser.add_argument('--lr', type=float, help='learning rate (default: 1e-3)', default=1e-3)
parser.add_argument('--momentum', type=float, help='momentum (default: 0.9)', default=0.9)
parser.add_argument('--wd', type=float, help='weight_decay, (default: 0)', default=0)
parser.add_argument('--network', type=str,
                    help='model type when using DUNs (other methods use ResNet) (default: ResNet)',
                    default='ResNet', choices=['ResNet', 'MLP'])
parser.add_argument('--n_runs', type=int, default="5",
                    help='number of runs with different random seeds to perform (default: 5)')
parser.add_argument('--n_queries', type=int, 
                    help='number of iterations for active learning (default: 20)', default=20)
parser.add_argument('--query_size', type=int, 
                    help='number of acquired data points in active learning (default: 10)',
                    default=10)
parser.add_argument('--query_strategy', choices=['random','entropy', 'variance'], 
                    help='type of acquisition function (default: random)', default='random')
parser.add_argument('--clip_var', action='store_true',
                    help='clip variance at 1 for variance acquisition (default: False)', default=False)
parser.add_argument('--sampling', action='store_true',
                    help='stochastic relaxation of acquisition function (default: False)', default=False)
parser.add_argument('--T', type=float, 
                    help='temperature for sampling acquisition (default: 1)', default=1)
parser.add_argument('--init_train', type=int, 
                    help='number of labelled observations in initial train set (default: 10)',
                    default=10)
parser.add_argument('--prior_decay', type=float, 
                    help='rate of decay for non-uniform prior distribution (default: None)',
                    default=None)
parser.add_argument('--bias_weights', action='store_true',
                    help='use active learning bias reduction weights in loss function (default: False)',
                    default=False)
parser.add_argument('--run_id', type=int, help='ID number of the active learning run')

args = parser.parse_args()

if args.bias_weights:
    args.sampling = True

# Some defaults
batch_size = args.batch_size
momentum = args.momentum
epochs = args.n_epochs
nb_its_dev = 5

name = '_'.join([args.inference, args.dataset, str(args.N_layers), str(args.width), str(args.lr), str(args.wd),
                 str(args.overcount), str(args.init_train)])
if args.network == 'MLP':
    name += '_MLP'
if args.prior_decay:
    name += f'_{args.prior_decay}'
if args.query_strategy != 'random':
    name += f'_{args.query_strategy}'
if args.clip_var:
    name += '_clip'
if args.sampling:
    name += f'_{args.T}T'
if args.bias_weights:
    name += '_biasw'
name += '_ntrain'
name += '_sampling'

cuda = (args.gpu is not None)
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('cuda', cuda)

mkdir(args.output_folder)

# Experiment runs
n_runs = args.n_runs
train_err = np.zeros(args.n_queries)
test_err = np.zeros(args.n_queries)
train_NLL = np.zeros(args.n_queries)
test_NLL = np.zeros(args.n_queries)
mll = np.zeros(args.n_queries)

j = int(args.run_id)
seed = j
mkdir(f'{args.output_folder}/{name}/{j}')

# Create datasets
if args.split==0:
    n_split = j%5 if args.dataset=='protein' else j%20
else:
    n_split = args.split
if args.dataset == "flights":
    X_train, X_test, _, _, y_train, y_test, y_means, y_stds = load_flight(base_dir=args.datadir,
                                                                            k800=(args.split == "800k"))
elif args.dataset in uci_names + uci_gap_names:
    gap = False
    if args.dataset in uci_gap_names:
        gap = True
        args.dataset = args.dataset[:-4]

    X_train, X_test, _, _, y_train, y_test, y_means, y_stds = \
        load_gap_UCI(base_dir=args.datadir, dname=args.dataset, n_split=int(n_split), gap=gap)

testset = Datafeed(torch.Tensor(X_test), torch.Tensor(y_test), transform=None)

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

valloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=False,
                                        num_workers=args.num_workers)
#testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
#                                        num_workers=args.num_workers)

trainset = DatafeedIndexed(torch.Tensor(X_train), torch.Tensor(y_train), args.init_train, seed=seed, transform=None)
n_labelled = int(sum(1 - trainset.unlabeled_mask))

# Active learning loop
for i in range(args.n_queries):

    query_results = {}

    # Instantiate model
    width = args.width
    n_layers = args.N_layers
    wd = args.wd
    lr = args.lr

    if args.inference == 'MFVI':
        prior_sig = 1

        model = MFVI_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers, prior_sig=1)

        net = regression_baseline_net_VI(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                        MC_samples=10, train_samples=5)

    elif args.inference == 'Dropout':
        model = dropout_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                        width=width, n_layers=n_layers, p_drop=0.1)

        net = regression_baseline_net(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                    MC_samples=10, weight_decay=wd)
    elif args.inference == 'SGD':
        model = SGD_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers)

        net = regression_baseline_net(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                    MC_samples=0, weight_decay=wd)
    elif args.inference == 'DUN':

        if args.network == 'ResNet':
            model = arq_uncert_fc_resnet(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
        elif args.network == 'MLP':
            model = arq_uncert_fc_MLP(input_dim, output_dim, width, n_layers, w_prior=None, BMA_prior=False)
        else:
            raise Exception('Bad network type. This should never raise as there is a previous assert.')

        if args.prior_decay:
            prior_probs = [(1 - args.prior_decay)**i for i in range(n_layers+1)]
            prior_probs = [p/sum(prior_probs) for p in prior_probs]
        else:
            prior_probs = [1 / (n_layers + 1)] * (n_layers + 1)
        print(f'prior dist: {prior_probs}')
        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None,
                        seed=seed, regression=True, pred_sig=None, weight_decay=wd)
    
    # Train model on labeled data
    labeled_idx = np.where(trainset.unlabeled_mask == 0)[0]
    labeledloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, num_workers=args.num_workers, sampler=torch.utils.data.SubsetRandomSampler(labeled_idx)
        )
    
    cprint('p', f'Query: {i}\tlabelled points: {len(labeled_idx)}')

    if args.inference in ['MFVI', 'Dropout', 'SGD']:
        marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
            approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
            train_fc_baseline(net, f'{name}/{j}', args.output_folder, batch_size, epochs, labeledloader, valloader, cuda=cuda,
                        flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None, track_posterior=False, 
                        track_exact_ELBO=False, seed=seed, save_freq=nb_its_dev, basedir_prefix=n_labelled,
                        bias_reduction_weights=args.bias_weights, dataset=trainset)
    else:
        marginal_loglike_estimate, train_mean_predictive_loglike, dev_mean_predictive_loglike, err_train, err_dev, \
            approx_d_posterior, true_d_posterior, true_likelihood, exact_ELBO, basedir = \
            train_fc_DUN(net, f'{name}/{j}', args.output_folder, batch_size, epochs, labeledloader, valloader,
                    cuda, seed=seed, flat_ims=False, nb_its_dev=nb_its_dev, early_stop=None,
                    track_posterior=True, track_exact_ELBO=False, tags=None,
                    load_path=None, save_freq=nb_its_dev, basedir_prefix=n_labelled, 
                    bias_reduction_weights=args.bias_weights, dataset=trainset)

    # Record performance (train err, test err, NLL)
    err_dev = err_dev[::nb_its_dev] # keep only epochs for which val error evaluated 
    dev_mean_predictive_loglike = dev_mean_predictive_loglike[::nb_its_dev]
    
    min_train_error = min([err for err in err_train if err>=0])
    min_test_error = min([err for err in err_dev if err>=0])
    min_train_nll = min([nll for nll in train_mean_predictive_loglike])
    min_test_nll = min([nll for nll in dev_mean_predictive_loglike])
    max_mll = max([mll for mll in marginal_loglike_estimate])
    train_err[i] = min_train_error
    test_err[i] = min_test_error
    train_NLL[i] = min_train_nll
    test_NLL[i] = min_test_nll
    mll[i] = max_mll

    # Save current dataset
    with open(f'{basedir}/media/trainset.pl', 'wb') as trainset_file:
        pl.dump(trainset, trainset_file)
    
    # Acquire data
    net.load(f'{basedir}/models/theta_best.dat')
    acquire_samples(net, trainset, args.query_size, query_strategy=args.query_strategy, 
                    clip_var=args.clip_var, sampling=args.sampling, T=args.T, seed=seed)
    n_labelled = int(sum(1 - trainset.unlabeled_mask))
    current_labeled_idx = np.where(trainset.unlabeled_mask == 0)[0]
    acquired_data_idx = current_labeled_idx[~np.isin(current_labeled_idx, labeled_idx)] 
    unlabeled_idx = np.concatenate([np.where(trainset.unlabeled_mask == 1)[0],acquired_data_idx])

    # Create plots
    media_dir = basedir + '/media'
    show_range = 5
    ylim = 3
    add_noise = False
    
    query_results['unlabeled_idx'] = unlabeled_idx.tolist()
    query_results['labeled_idx'] = labeled_idx.tolist()
    query_results['acquired_data_idx'] = acquired_data_idx.tolist()
    #np.savetxt(f'{media_dir}/unlabeled_idx.csv', unlabeled_idx, delimiter=',')
    #np.savetxt(f'{media_dir}/labeled_idx.csv', labeled_idx, delimiter=',')
    #np.savetxt(f'{media_dir}/acquired_data_idx.csv', acquired_data_idx, delimiter=',')

    if args.inference == 'DUN':
        query_results['approx_d_posterior'] = approx_d_posterior[-1,:].tolist()
        query_results['true_d_posterior'] = true_d_posterior[-1,:].tolist()
        #np.savetxt(f'{media_dir}/approx_d_posterior.csv', approx_d_posterior, delimiter=',')
        #np.savetxt(f'{media_dir}/true_d_posterior.csv', true_d_posterior, delimiter=',')
    
    with open(f'{media_dir}/query_results.json', 'w') as outfile:
        json.dump(query_results, outfile)

cprint('p', f'Train errors: {train_err}')
cprint('p', f'Val errors: {test_err}\n')
#np.savetxt(f'{args.output_folder}/{name}/{j}/train_err_{j}.csv', train_err, delimiter=',')
#np.savetxt(f'{args.output_folder}/{name}/{j}/test_err_{j}.csv', test_err, delimiter=',')
#np.savetxt(f'{args.output_folder}/{name}/{j}/train_nll_{j}.csv', train_NLL, delimiter=',')
#np.savetxt(f'{args.output_folder}/{name}/{j}/test_nll_{j}.csv', test_NLL, delimiter=',')
#np.savetxt(f'{args.output_folder}/{name}/{j}/mll_{j}.csv', mll, delimiter=',')

results = {
    'train_err': train_err.tolist(),
    'test_err': test_err.tolist(),
    'train_nll': train_NLL.tolist(),
    'test_nll': test_NLL.tolist(),
    'mll': mll.tolist()
}
with open(f'{args.output_folder}/{name}/{j}/results_{j}.json', 'w') as outfile:
    json.dump(results, outfile)

toc = time()
cprint('r', f'Time: {toc - tic}')