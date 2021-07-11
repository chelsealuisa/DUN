import numpy as np
from src.plots import plot_al_mean_rmse, plot_al_results
import matplotlib.pyplot as plt
import os
import fnmatch
import pickle as pl
from src.DUN.training_wrappers import DUN_VI
from src.DUN.stochastic_fc_models import arq_uncert_fc_resnet
from src.probability import depth_categorical_VI
from src.utils import cprint
import torch

c = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd',
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#d62728']


def plot_rmse_mean_std_single_method():
    # Mean and std per method
    files = [
        #'SGD_wiggle_3_100_0.001_0.0001_1_10_110621',
        'DUN_agw_1d_10_100_0.001_0.0001_1_10_110621',
        'DUN_matern_1d_10_100_0.001_0.0001_1_10_110621',
        'DUN_my_1d_10_100_0.001_0.0001_1_10_110621',
        ]

    savedir = 'saves'
    init_train_size = 10
    query_size = 10
    for file in files:
        results = np.genfromtxt(f'{savedir}/{file}/results.csv', delimiter=',')
        means = results[:,0]
        stds = results[:,1]
        if file.split('_')[1]=='wiggle':
            n_queries = 20
        elif file.split('_')[1]=='andrew':
            n_queries = 15
            query_size = 5
        else:
            n_queries = 30
        plot_al_results(f'{savedir}/{file}/rmse_plot', file, means, stds, n_queries, query_size, init_train_size, ylog=False)


def plot_mean_rmse_all_methods():
    # Mean of all methods
    dataset = 'wiggle'
    savedir = 'saves'
    if dataset=='wiggle':
        n_queries = 20
    else:
        n_queries = 30
    query_size = 10
    init_train = 10
    dun = np.genfromtxt(f'{savedir}/DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_ntrain/results.csv', delimiter=',')[:,0]
    dropout = np.genfromtxt(f'{savedir}/Dropout_{dataset}_3_100_0.001_0.0001_1_{init_train}_ntrain/results.csv', delimiter=',')[:,0]
    mfvi = np.genfromtxt(f'{savedir}/MFVI_{dataset}_3_100_0.001_0.0001_1_{init_train}_ntrain/results.csv', delimiter=',')[:,0]
    sgd = np.genfromtxt(f'{savedir}/SGD_{dataset}_3_100_0.001_0.0001_1_{init_train}_ntrain/results.csv', delimiter=',')[:,0]
    plot_al_mean_rmse(f'{savedir}/{dataset}_rmse_all_random', dun, dropout, mfvi, sgd, dataset, n_queries, query_size, init_train)


def plot_rmse_errorbars():
    # Compare RMSE with error bars
    dataset = 'wine'
    savedir = 'saves_regression'
    n_queries = 10
    init_train_size = 20
    query_size = 1

    file_1 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_ntrain'
    file_2 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_variance_ntrain'
    file_3 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
    #file_4 = f'SGD_{dataset}_3_100_0.001_0.0001_1_{init_train_size}_ntrain'
    results_1 = np.genfromtxt(f'{savedir}/{file_1}/test_err.csv', delimiter=',')
    results_2 = np.genfromtxt(f'{savedir}/{file_2}/test_err.csv', delimiter=',')
    results_3 = np.genfromtxt(f'{savedir}/{file_3}/test_err.csv', delimiter=',')
    #results_4 = np.genfromtxt(f'{savedir}/{file_4}/test_err.csv', delimiter=',')
    means_1 = results_1[:,0]
    stds_1 = results_1[:,1]
    means_2 = results_2[:,0]
    stds_2 = results_2[:,1]
    means_3 = results_3[:,0]
    stds_3 = results_3[:,1]
    #means_4 = results_4[:,0]
    #stds_4 = results_4[:,1]

    plt.figure(dpi=300)
    x = np.arange(init_train_size, init_train_size + n_queries*query_size, query_size)
    plt.plot(x, means_1, c='tab:blue', label='Random')
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color='tab:blue')
    plt.plot(x, means_2, c='tab:red', label='Max variance')
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color='tab:red')
    plt.plot(x, means_3, c='tab:green', label='Max variance, clipped')
    plt.fill_between(x, means_3+stds_3, means_3-stds_3, alpha=0.3, color='tab:green')
    #plt.plot(x, means_4, c='tab:orange', label='SGD')
    #plt.fill_between(x, means_4+stds_4, means_4-stds_4, alpha=0.3, color='tab:orange')
    plt.xlabel('Train set size')
    plt.ylabel('Validation RMSE')
    plt.title(f'{dataset} dataset')
    plt.legend()
    plt.savefig( f'{savedir}/{dataset}_DUN_miguel.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def plot_rmse_errorbars_80runs():
    # Compare RMSE with error bars
    dataset = 'wine'
    savedir = 'saves_regression'
    n_queries = 10
    init_train_size = 20
    query_size = 1

    file_1 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_ntrain'
    file_2 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_variance_ntrain'
    file_3 = f'DUN_{dataset}_5_10_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
    results_1a = np.genfromtxt(f'{savedir}/{file_1}/test_err_0-39.csv', delimiter=',')[:,2:]
    results_1b = np.genfromtxt(f'{savedir}/{file_1}/test_err_40-79.csv', delimiter=',')[:,2:]
    results_2a = np.genfromtxt(f'{savedir}/{file_2}/test_err_0-39.csv', delimiter=',')[:,2:]
    results_2b = np.genfromtxt(f'{savedir}/{file_2}/test_err_40-79.csv', delimiter=',')[:,2:]
    results_3a = np.genfromtxt(f'{savedir}/{file_3}/test_err_0-39.csv', delimiter=',')[:,2:]
    results_3b = np.genfromtxt(f'{savedir}/{file_3}/test_err_40-79.csv', delimiter=',')[:,2:]
    
    results_1 = np.concatenate([results_1a, results_1b], axis=1)
    results_2 = np.concatenate([results_2a, results_2b], axis=1)
    results_3 = np.concatenate([results_3a, results_3b], axis=1)

    means_1 = np.mean(results_1, axis=1)
    stds_1 = np.std(results_1, axis=1)
    means_2 = np.mean(results_2, axis=1)
    stds_2 = np.std(results_2, axis=1)
    means_3 = np.mean(results_3, axis=1)
    stds_3 = np.std(results_3, axis=1)

    plt.figure(dpi=300)
    x = np.arange(init_train_size, init_train_size + n_queries*query_size, query_size)
    plt.plot(x, means_1, c='tab:blue', label='Random')
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color='tab:blue')
    plt.plot(x, means_2, c='tab:red', label='Max variance')
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color='tab:red')
    plt.plot(x, means_3, c='tab:green', label='Max variance, clipped')
    plt.fill_between(x, means_3+stds_3, means_3-stds_3, alpha=0.3, color='tab:green')
    plt.xlabel('Train set size')
    plt.ylabel('Validation RMSE')
    plt.title(f'{dataset} dataset')
    plt.legend()
    plt.savefig( f'{savedir}/{dataset}_DUN_miguel_80runs.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def plot_sidebyside_posterior_barplots():
    # Side-by-side posterior bar plots
    dir = './saves_regression/DUN_naval_10_100_0.001_0.0001_1_50_variance_clip_ntrain'
    small = 50
    large = 630
    n_runs = 5

    posteriors = []
    for i in range(n_runs):
        for file in os.listdir(f'{dir}/{i}/'):
            if fnmatch.fnmatch(file, f'{small}_*'):
                f_name = file
        posteriors.append(np.genfromtxt(f'{dir}/{i}/{f_name}/media/approx_d_posterior.csv', delimiter=',')[-1,:])
    posteriors = np.array(posteriors)
    means_small = posteriors.mean(axis=0)
    stds_small = posteriors.std(axis=0)

    posteriors = []
    for i in range(n_runs):
        for file in os.listdir(f'{dir}/{i}/'):
            if fnmatch.fnmatch(file, f'{large}_*'):
                f_name = file
        posteriors.append(np.genfromtxt(f'{dir}/{i}/{f_name}/media/approx_d_posterior.csv', delimiter=',')[-1,:])
    posteriors = np.array(posteriors)
    means_large = posteriors.mean(axis=0)
    stds_large = posteriors.std(axis=0)
        
    x = np.array([i for i in range(means_small.shape[0])])
    width = 0.4 # width of bars

    fig, ax = plt.subplots(dpi=300)
    ax1 = ax.bar(x - width/2, means_small, width, yerr=stds_small, label=f'Train size: {small}')
    ax2 = ax.bar(x + width/2, means_large, width, yerr=stds_large, label=f'Train size: {large}')
    plt.title(f'Approx posterior distribution over depth')
    plt.xlabel('Layer')
    plt.legend(loc='upper left')
    plt.savefig(f'{dir}/posterior_plots/{small}_{large}_d_post_approx.pdf', format='pdf', bbox_inches='tight')
    with open(f'{dir}/posterior_plots/{small}_{large}_depth_post_approx.pickle', 'wb') as output_file:
        pl.dump(fig, output_file)
    plt.close(fig=None)


def plot_bias_reduction_weights():
    # Bias reduction weights plots
    savedir = 'saves_regression'
    dataset = 'boston'
    n_queries = 17
    init_train_size = 20
    query_size = 20
    method = 'MFVI'
    depth = 3

    dir_weighted = f'{method}_{dataset}_{depth}_100_0.001_0.0001_1_{init_train_size}_variance_clip_biasw_ntrain'
    dir_unweighted = f'{method}_{dataset}_{depth}_100_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain_nll'
    dir_unbiased = f'{method}_{dataset}_{depth}_100_0.001_0.0001_1_ntrain_noAL'

    nll_unbiased = np.genfromtxt(f'{savedir}/{dir_unbiased}/results_NLL.csv', delimiter=',')
    nll_weighted_mean = np.genfromtxt(f'{savedir}/{dir_weighted}/results_NLL.csv', delimiter=',')[:,0]
    nll_weighted_std = np.genfromtxt(f'{savedir}/{dir_weighted}/results_NLL.csv', delimiter=',')[:,1]
    nll_unweighted_mean = np.genfromtxt(f'{savedir}/{dir_unweighted}/results_NLL.csv', delimiter=',')[:,0]
    nll_unweighted_std = np.genfromtxt(f'{savedir}/{dir_unweighted}/results_NLL.csv', delimiter=',')[:,1]

    err_weighted_mean = np.genfromtxt(f'{savedir}/{dir_weighted}/results.csv', delimiter=',')[:,0]
    err_weighted_std = np.genfromtxt(f'{savedir}/{dir_weighted}/results.csv', delimiter=',')[:,1]
    err_unweighted_mean = np.genfromtxt(f'{savedir}/{dir_unweighted}/results.csv', delimiter=',')[:,0]
    err_unweighted_std = np.genfromtxt(f'{savedir}/{dir_unweighted}/results.csv', delimiter=',')[:,1]

    bias_weighted = nll_unbiased - nll_weighted_mean
    bias_unweighted = nll_unbiased - nll_unweighted_mean

    # plot bias
    plt.figure(dpi=300)
    x = np.arange(init_train_size, init_train_size + n_queries*query_size, query_size)
    plt.plot(x, bias_unweighted, c='tab:blue', label='r - E[R]')
    plt.fill_between(x, bias_unweighted+nll_unweighted_std, bias_unweighted-nll_unweighted_std, alpha=0.3, color='tab:blue')
    plt.plot(x, bias_weighted, c='tab:red', label='r - E[R_LURE]')
    plt.fill_between(x, bias_weighted+nll_weighted_std, bias_weighted-nll_weighted_std, alpha=0.3, color='tab:red')
    plt.xlabel('Train set size')
    plt.ylabel('Bias: r - E[*]')
    plt.title(f'{dataset} dataset')
    plt.legend()
    plt.savefig( f'{savedir}/{dataset}_{method}_bias_weights_bias.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # plot error
    plt.figure(dpi=300)
    x = np.arange(init_train_size, init_train_size + n_queries*query_size, query_size)
    plt.plot(x, err_unweighted_mean, c='tab:blue', label='Trained with R')
    plt.fill_between(x, err_unweighted_mean+err_unweighted_std, err_unweighted_mean-err_unweighted_std, alpha=0.3, color='tab:blue')
    plt.plot(x, err_weighted_mean, c='tab:red', label='Trained with R_LURE')
    plt.fill_between(x, err_weighted_mean+err_weighted_std, err_weighted_mean-err_weighted_std, alpha=0.3, color='tab:red')
    plt.xlabel('Train set size')
    plt.ylabel('Validation RMSE')
    plt.title(f'{dataset} dataset')
    plt.legend()
    plt.savefig( f'{savedir}/{dataset}_{method}_bias_weights_error.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def plot_mean_layerwise():
    basedir = 'saves/DUN_wiggle_10_100_0.001_0.0001_1_10_variance_clip_ntrain'
    run = 0
    acq_step = '10_ayxp00me'
    inference = 'DUN'
    query_strategy = 'variance'
    clip_var = True
    media_dir = f'{basedir}/{run}/{acq_step}/media/'
    
    X_train = np.genfromtxt(f'{basedir}/X_train.csv', delimiter=',')
    y_train = np.genfromtxt(f'{basedir}/y_train.csv', delimiter=',')
    labeled_idx = np.genfromtxt(f'{basedir}/{run}/{acq_step}/media/labeled_idx.csv', delimiter=',').astype('int')
    acquired_data_idx = np.genfromtxt(f'{basedir}/{run}/{acq_step}/media/acquired_data_idx.csv', delimiter=',').astype('int')
    unlabeled_idx = np.genfromtxt(f'{basedir}/{run}/{acq_step}/media/unlabeled_idx.csv', delimiter=',').astype('int')
    input_dim = 1
    output_dim = 1

    n_labelled = len(labeled_idx)
    model = arq_uncert_fc_resnet(input_dim, output_dim, 100, 10, w_prior=None, BMA_prior=False)
    prior_probs = [1 / (10 + 1)] * (10 + 1)
    prob_model = depth_categorical_VI(prior_probs, cuda=None)
    net = DUN_VI(model, prob_model, n_labelled, lr=0.001, momentum=0.9, cuda=None, schedule=None,
                regression=True, pred_sig=None, weight_decay=0.0001)
    
    net.load(f'{basedir}/{run}/{acq_step}/models/theta_best.dat')

    show_range = 5
    ylim = 3
    x_view = np.linspace(-show_range, show_range, 8000)
    x_view = torch.Tensor(x_view).unsqueeze(1)

    # Layerwise predictions mean and std dev
    if inference != 'DUN':
        pred_mu, pred_std = net.predict(x_view, Nsamples=50, return_model_std=True)
    else:
        pred_mu, pred_std = net.predict(x_view, get_std=True, return_model_std=True)
    pred_mu = pred_mu.data.cpu().numpy()
    pred_std = pred_std.data.cpu().numpy()
    noise_std = net.f_neg_loglike.log_std.exp().data.cpu().numpy()

    if query_strategy=='variance':
        fig_handle = plt.figure(dpi=300)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.scatter(X_train[unlabeled_idx], y_train[unlabeled_idx], s=3, alpha=0.2, c=c[0])
        ax1.scatter(X_train[labeled_idx], y_train[labeled_idx], s=5, alpha=0.7, c='k')
        ax1.plot(x_view, pred_mu, c=c[3])
        ax1.fill_between(x_view[:,0], 
                        pred_mu[:,0] + (pred_std[:,0]**2 + noise_std**2)**0.5, 
                        pred_mu[:,0] - (pred_std[:,0]**2 + noise_std**2)**0.5, 
                        alpha=0.2, color=c[3])
        ax1.scatter(X_train[acquired_data_idx], y_train[acquired_data_idx], s=5, alpha=0.7, c=c[2])
        ax1.set_title('Mean predictive function')
        ax1.set_ylim([-ylim, ylim])
        ax1.set_xlim([-show_range, show_range])
        plt.tight_layout()

        ax2.plot(x_view, pred_std, c='k', linewidth=1)
        if clip_var:
            pred_std = np.where(pred_std>1, 1, pred_std)
            ax2.plot(x_view, pred_std, c='b', linestyle='--', linewidth=1.5)
        plt.yscale('log')
        ax2.set_title('Acquisition function')
        ax2.set_xlim([-show_range, show_range])
        plt.tight_layout()

        fig.savefig(f'{media_dir}/mean_layerwise_TEST.pdf', format='pdf', bbox_inches='tight')
        with open(f'{media_dir}/mean_layerwise_TEST.pickle', 'wb') as output_file:
            pl.dump(fig_handle, output_file)
        plt.close('all')

    else:
        fig_handle = plt.figure(dpi=300)
        plt.scatter(X_train[unlabeled_idx], y_train[unlabeled_idx], s=3, alpha=0.2, c=c[0])
        plt.scatter(X_train[labeled_idx], y_train[labeled_idx], s=5, alpha=0.7, c='k')
        plt.plot(x_view, pred_mu, c=c[3])
        plt.fill_between(x_view[:,0], 
                        pred_mu[:,0] + (pred_std[:,0]**2 + noise_std**2)**0.5, 
                        pred_mu[:,0] - (pred_std[:,0]**2 + noise_std**2)**0.5, 
                        alpha=0.2, color=c[3])
        plt.scatter(X_train[acquired_data_idx], y_train[acquired_data_idx], s=5, alpha=0.7, c=c[2])
        plt.title('Mean predictive function')
        plt.ylim([-ylim, ylim])
        plt.xlim([-show_range, show_range])
        plt.tight_layout()
        plt.savefig(f'{media_dir}/mean_layerwise_TEST.pdf', format='pdf', bbox_inches='tight')
        with open(f'{media_dir}/mean_layerwise_TEST.pickle', 'wb') as output_file:
            pl.dump(fig_handle, output_file)
        plt.close('all')


if __name__=="__main__":
    #plot_bias_reduction_weights()
    plot_rmse_errorbars_80runs()
    #plot_mean_layerwise()
    #plot_sidebyside_posterior_barplots()