import numpy as np
from src.plots import plot_al_mean_rmse, plot_al_rmse
import matplotlib.pyplot as plt
import os
import fnmatch
import pickle as pl


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
        plot_al_rmse(f'{savedir}/{file}/rmse_plot', file, means, stds, n_queries, query_size, init_train_size, ylog=False)


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
    dataset = 'boston'
    savedir = 'saves_regression'
    n_queries = 17
    init_train_size = 20
    query_size = 20

    file_1 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
    file_2 = f'SGD_{dataset}_1_100_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
    file_3 = f'SGD_{dataset}_4_100_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
    #file_4 = f'SGD_{dataset}_3_100_0.001_0.0001_1_{init_train_size}_ntrain'
    results_1 = np.genfromtxt(f'{savedir}/{file_1}/results.csv', delimiter=',')
    results_2 = np.genfromtxt(f'{savedir}/{file_2}/results.csv', delimiter=',')
    results_3 = np.genfromtxt(f'{savedir}/{file_3}/results.csv', delimiter=',')
    #results_4 = np.genfromtxt(f'{savedir}/{file_4}/results.csv', delimiter=',')
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
    plt.plot(x, means_1, c='tab:blue', label='DUN')
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color='tab:blue')
    plt.plot(x, means_2, c='tab:red', label='SGD, depth 1')
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color='tab:red')
    plt.plot(x, means_3, c='tab:orange', label='SGD, depth 4')
    plt.fill_between(x, means_3+stds_3, means_3-stds_3, alpha=0.3, color='tab:orange')
    #plt.plot(x, means_4, c='tab:orange', label='SGD')
    #plt.fill_between(x, means_4+stds_4, means_4-stds_4, alpha=0.3, color='tab:orange')
    plt.xlabel('Train set size')
    plt.ylabel('Validation RMSE')
    plt.title(f'{dataset} dataset')
    plt.legend()
    plt.savefig( f'{savedir}/{dataset}_DUN_SGD_fixed_depth.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def plot_sidebyside_posterior_barplots():
    # Side-by-side posterior bar plots
    dir = './saves_regression/DUN_concrete_10_100_0.001_0.0001_1_50_variance_clip_ntrain'
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
    plt.legend(loc='lower right')
    plt.savefig(f'{dir}/posterior_plots/{small}_{large}_d_post_approx.pdf', format='pdf', bbox_inches='tight')
    with open(f'{dir}/posterior_plots/{small}_{large}_depth_post_approx.pickle', 'wb') as output_file:
        pl.dump(fig, output_file)
    plt.close(fig=None)


def plot_bias_reduction_weights():
    # Bias reduction weights plots
    savedir = 'saves'
    dataset = 'wiggle'
    n_queries = 20
    init_train_size = 10
    query_size = 10
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


if __name__=="__main__":
    plot_bias_reduction_weights()