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
from torch.utils.data import dataset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

from src.datasets.image_loaders import get_image_loader
from src.utils import DatafeedImage, DatafeedImageIndexed, cprint, mkdir
from src.probability import depth_categorical_VI
from src.DUN.train_fc import train_fc_DUN
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_img_resnets import resnet18, resnet34, resnet50, resnet101, arq_uncert_conv2d_resnet
from src.baselines.SGD import SGD_regression_homo
from src.baselines.dropout import dropout_regression_homo
from src.baselines.mfvi import MFVI_regression_homo
from src.baselines.training_wrappers import regression_baseline_net, regression_baseline_net_VI
from src.baselines.train_fc import train_fc_baseline
from src.acquisition_fns import acquire_samples

matplotlib.use('Agg')

tic = time()

parser = argparse.ArgumentParser(description='Classification dataset running script')

parser.add_argument('--n_epochs', default=None, type=int,
                    help='number of total epochs to run (if None, use dataset default)')
parser.add_argument('--dataset', help='which dataset to train on (default: MNIST)', default='MNIST',
                    choices=['MNIST'])
parser.add_argument('--model', type=str, default='resnet50',
                    choices=["arq_uncert_conv2d_resnet", "resnet18", "resnet32", "resnet50", "resnet101"],
                    help='model to train (default: resnet50)')
parser.add_argument('--start_depth', default=1, type=int,
                    help='first layer to be uncertain about (default: 1)')
parser.add_argument('--end_depth', default=13, type=int,
                    help='last layer to be uncertain about + 1 (default: 13)')
parser.add_argument('--q_nograd_its', default=0, type=int,
                    help='number of warmup epochs (where q is not learnt) (default: 0)')
parser.add_argument('--data_dir', type=str, help='where to save dataset (default: ../data/)',
                    default='../data/')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use. (default: 0)')
parser.add_argument('--inference', type=str, help='model to use (default: DUN)',
                    default='DUN', choices=['DUN', 'MFVI', 'Dropout', 'SGD'])
parser.add_argument('--num_workers', type=int, help='number of parallel workers for dataloading (default: 4)', default=4)
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size to use. (default: 64)')
parser.add_argument('--output_folder', type=str, help='where to save results (default: ./saves_images/)',
                    default='./saves_images/')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--n_runs', type=int, default='5',
                    help='number of runs with different random seeds to perform (default: 5)')
parser.add_argument('--n_queries', type=int, 
                    help='number of iterations for active learning (default: 10)', default=10)
parser.add_argument('--query_size', type=int, 
                    help='number of acquired data points in active learning (pdefault: 10)',
                    default=10)
parser.add_argument('--query_strategy', choices=['random','entropy', 'variance'], 
                    help='type of acquisition function (default: random)', default='random')
parser.add_argument('--sampling', action='store_true',
                    help='stochastic relaxation of acquisition function (default: False)', default=False)
parser.add_argument('--T', type=int, 
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

epoch_dict = {
    'Imagenet': 90,
    'SmallImagenet': 90,
    'CIFAR10': 300,
    'CIFAR100': 300,
    'SVHN': 90,
    'Fashion': 90,
    'MNIST': 90
}
milestone_dict = {
    'Imagenet': [30, 60],  # This is pytorch default
    'SmallImagenet': [30, 60],
    'CIFAR10': [150, 225],
    'CIFAR100': [150, 225],
    'SVHN': [50, 70],
    'Fashion': [40, 70],
    'MNIST': [40, 70]
}

# Defaults
best_err1 = 1
lr = 0.1
momentum = 0.9
wd = 1e-4
nb_its_dev = 5

data_set = args.dataset
workers = args.num_workers
epochs = args.n_epochs
resume = args.resume
output_folder = args.output_folder
q_nograd_its = args.q_nograd_its
batch_size = args.batch_size
data_dir = args.data_dir
start_depth = args.start_depth
end_depth = args.end_depth
model = args.model
n_layers = end_depth - start_depth
if epochs is None:
    epochs = epoch_dict[args.dataset]
milestones = milestone_dict[args.dataset]

initial_conv = '3x3' if data_set in ['Imagenet', 'SmallImagenet'] else '1x3'
input_chanels = 1 if data_set in ['MNIST', 'Fashion'] else 3
if data_set in ['Imagenet', 'SmallImagenet']:
    num_classes = 1000
elif data_set in ['CIFAR100']:
    num_classes = 100
else:
    num_classes = 10

if model == 'resnet18':
    model_class = resnet18
elif model == 'resnet18':
    model_class = resnet34
elif model == 'resnet50':
    model_class = resnet50
elif model == 'resnet101':
    model_class = resnet101
elif model == 'arq_uncert_conv2d_resnet':
    model_class = arq_uncert_conv2d_resnet
else:
    raise Exception('requested model not implemented')

cuda = (args.gpu is not None)
if cuda:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
print('cuda', cuda)

name = '_'.join([args.inference, args.dataset, model, str(n_layers), f"warm{q_nograd_its}", str(lr), str(wd), str(args.init_train)])
if args.prior_decay:
    name += f'_{args.prior_decay}'
if args.query_strategy != 'random':
    name += f'_{args.query_strategy}'
if args.sampling:
    name += f'_{args.T}T'

mkdir(args.output_folder)

# Create datasets
if data_set == 'MNIST':
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])

    train_dataset = datasets.MNIST(root=args.data_dir, train=True, download=True,
                                    transform=transform_train)
    test_dataset = datasets.MNIST(root=args.data_dir, train=False, download=True,
                                    transform=transform_test)
    input_dim = 1
    output_dim = 10

    X_train = train_dataset.data
    X_test = test_dataset.data
    y_train = train_dataset.targets
    y_test = test_dataset.targets

testset = DatafeedImage(X_test, y_test, transform=transform_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

valloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True,
                                        num_workers=args.num_workers)

# Experiment runs
train_err = np.zeros(args.n_queries)
test_err = np.zeros(args.n_queries)
train_NLL = np.zeros(args.n_queries)
test_NLL = np.zeros(args.n_queries)
mll = np.zeros(args.n_queries)

j = args.run_id
seed = j
mkdir(f'{args.output_folder}/{name}/{j}')

# Reset train data
trainset = DatafeedImageIndexed(X_train, y_train, args.init_train, 
                                seed=j, transform=transform_train)
n_labelled = int(sum(1 - trainset.unlabeled_mask))

# Active learning loop
for i in range(args.n_queries):

    query_results = {}

    width = 50 # TODO: remove once baseline method code in place

    # Instantiate model
    if args.inference == 'MFVI':
        prior_sig = 1

        model = MFVI_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers, prior_sig=1)

        net = regression_baseline_net_VI(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                        MC_samples=10, train_samples=5, regression=False)

    elif args.inference == 'Dropout':
        model = dropout_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                        width=width, n_layers=n_layers, p_drop=0.1)

        net = regression_baseline_net(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                    MC_samples=10, weight_decay=wd, regression=False)
    elif args.inference == 'SGD':
        model = SGD_regression_homo(input_dim=input_dim, output_dim=output_dim,
                                    width=width, n_layers=n_layers)

        net = regression_baseline_net(model, n_labelled, lr=args.lr, momentum=momentum, cuda=cuda, schedule=None, seed=seed,
                                    MC_samples=0, weight_decay=wd, regression=False)
    elif args.inference == 'DUN':

        if model_class==arq_uncert_conv2d_resnet:
            model = model_class(input_chan=input_chanels, output_dim=num_classes, outer_width=64, 
                        inner_width=32, n_layers=n_layers)
        else:
            model = model_class(arch_uncert=True, start_depth=start_depth, end_depth=end_depth, num_classes=num_classes,
                        zero_init_residual=True, initial_conv=initial_conv, concat_pool=False,
                        input_chanels=input_chanels, p_drop=0)

        prior_probs = [1 / (n_layers+1)] * (n_layers+1)
        prob_model = depth_categorical_VI(prior_probs, cuda=cuda)
        net = DUN_VI(model, prob_model, n_labelled, lr=lr, momentum=momentum, cuda=cuda, schedule=milestones,
                    seed=seed, regression=False, pred_sig=None, weight_decay=wd)
    
    # Train model on labeled data
    labeled_idx = np.where(trainset.unlabeled_mask == 0)[0]
    labeledloader = torch.utils.data.DataLoader(
        trainset, batch_size, num_workers=args.num_workers, sampler=torch.utils.data.SubsetRandomSampler(labeled_idx)
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
    
    # Acquire data
    net.load(f'{basedir}/models/theta_best.dat')
    acquire_samples(net, trainset, args.query_size, query_strategy=args.query_strategy,
                    clip_var=False, sampling=args.sampling, T=args.T, seed=seed)
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
#np.savetxt(f'{args.output_folder}/{name}/{j}/train_err_{j}.csv', train_err[:,j], delimiter=',')
#np.savetxt(f'{args.output_folder}/{name}/{j}/test_err_{j}.csv', test_err[:,j], delimiter=',')

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
cprint('r', toc - tic)