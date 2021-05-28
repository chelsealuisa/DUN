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
from src.datasets.additional_gap_loader import load_my_1d, load_agw_1d, load_andrew_1d
from src.datasets.additional_gap_loader import load_matern_1d, load_axis, load_origin, load_wiggle
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

c = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']

parser = argparse.ArgumentParser(description='Toy dataset running script')

parser.add_argument('--n_epochs', type=int, default=4000,
                    help='number of iterations performed by the optimizer (default: 4000)')
parser.add_argument('--dataset', help='toggles which dataset to optimize for (default: my_1d)',
                    choices=['my_1d', 'matern_1d', 'agw_1d', 'andrew_1d', 'wiggle'], default='my_1d')
parser.add_argument('--datadir', type=str, help='where to save dataset (default: ../data/)',
                    default='../data/')
parser.add_argument('--gpu', type=int, help='which GPU to run on (default: None)', default=None)
parser.add_argument('--inference', type=str, help='model to use (default: DUN)',
                    default='DUN', choices=['DUN', 'MFVI', 'Dropout', 'SGD'])
parser.add_argument('--num_workers', type=int, help='number of parallel workers for dataloading (default: 1)', default=1)
parser.add_argument('--N_layers', type=int, help='number of hidden layers to use (default: 2)', default=2)
parser.add_argument('--width', type=int, help='number of hidden units to use (default: 50)', default=50)
parser.add_argument('--savedir', type=str, help='where to save results (default: ./saves/)',
                    default='./saves/')
parser.add_argument('--overcount', type=int, help='how many times to count data towards ELBO (default: 1)', default=1)
parser.add_argument('--lr', type=float, help='learning rate (default: 1e-3)', default=1e-3)
parser.add_argument('--wd', type=float, help='weight_decay, (default: 0)', default=0)
parser.add_argument('--network', type=str,
                    help='model type when using DUNs (other methods use ResNet) (default: ResNet)',
                    default='ResNet', choices=['ResNet', 'MLP'])
parser.add_argument('--n_queries', type=int, 
                    help='number of iterations for active learning (default: 20)', default=20)
parser.add_argument('--query_size', type=int, 
                    help='number of acquired data points in active learning (default: 10)',
                    default=10)

args = parser.parse_args()


# Some defaults
batch_size = 2048
momentum = 0.9
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
if args.dataset == 'my_1d':
    X_train, y_train, X_test, y_test = load_my_1d(args.datadir)
elif args.dataset == 'matern_1d':
    X_train, y_train = load_matern_1d(args.datadir)
elif args.dataset == 'agw_1d':
    X_train, y_train = load_agw_1d(args.datadir, get_feats=False)
elif args.dataset == 'andrew_1d':
    X_train, y_train = load_andrew_1d(args.datadir)
elif args.dataset == 'axis':
    X_train, y_train = load_axis(args.datadir)
elif args.dataset == 'origin':
    X_train, y_train = load_origin(args.datadir)
elif args.dataset == 'wiggle':
    X_train, y_train = load_wiggle()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)

trainset = DatafeedIndexed(torch.Tensor(X_train), torch.Tensor(y_train), transform=None)
valset = Datafeed(torch.Tensor(X_val), torch.Tensor(y_val), transform=None)
testset = Datafeed(torch.Tensor(X_test), torch.Tensor(y_test), transform=None)
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                        num_workers=args.num_workers)

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
    N_train = X_train.shape[0]
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

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

            x_view = np.linspace(-show_range, show_range, 8000)
            subsample = 1
            x_view = torch.Tensor(x_view).unsqueeze(1)
            layer_preds = net.layer_predict(x_view).data.cpu().numpy()

            # Layerwise predictive functions (separate image)
            for i in range(layer_preds.shape[0]):
                plt.figure(dpi=80)
                plt.scatter(X_train[trainset.unlabeled_mask.astype(bool)], y_train[trainset.unlabeled_mask.astype(bool)], s=3, alpha=0.2, c=c[0])
                plt.scatter(X_train[~trainset.unlabeled_mask.astype(bool)], y_train[~trainset.unlabeled_mask.astype(bool)], s=5, alpha=0.7, c='k')
                _ = plt.plot(x_view[:, 0], layer_preds[i, :, 0].T, alpha=0.8, c='r')
                plt.title(i)
                plt.ylim([-ylim, ylim])
                plt.xlim([-show_range, show_range])
                plt.tight_layout()
                plt.savefig(f'{media_dir}/{str(i)}_layerwise.png', format='png', bbox_inches='tight')
                plt.close()

            # Layerwise predictive functions (single image)
            plt.figure(dpi=80)
            plt.scatter(X_train[trainset.unlabeled_mask.astype(bool)], y_train[trainset.unlabeled_mask.astype(bool)], s=3, alpha=0.2, c=c[0])
            plt.scatter(X_train[~trainset.unlabeled_mask.astype(bool)], y_train[~trainset.unlabeled_mask.astype(bool)], s=5, alpha=0.7, c='k')
            for i in range(layer_preds.shape[0]):
                plt.plot(x_view[:, 0], layer_preds[i, :, 0].T, alpha=0.8, c=c[i])
            plt.title('Layerwise predictive functions')
            plt.ylim([-ylim, ylim])
            plt.xlim([-show_range, show_range])
            plt.tight_layout()
            plt.savefig(f'{media_dir}/layerwise.png', format='png', bbox_inches='tight')
            plt.close()

            # Posterior over depth
            x = np.array([i for i in range(layer_preds.shape[0])])
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

    # plot validation error
    plt.figure(dpi=200)
    plt.plot(results[:,j])
    plt.xlabel('Query number')
    plt.ylabel('Validation error')
    plt.tight_layout()
    plt.savefig(f'{args.savedir}/{name}/{j}/val_error.png', format='png', bbox_inches='tight')

means = results.mean(axis=1).reshape(-1,1)
stds = results.std(axis = 1).reshape(-1,1)
results = np.concatenate((means, stds, results), axis=1)
np.savetxt(f'{args.savedir}/{name}/results.csv', results, delimiter=',')

toc = time()
cprint('r', toc - tic)