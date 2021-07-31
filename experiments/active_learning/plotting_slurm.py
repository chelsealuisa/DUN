import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from src.utils import mkdir
import os
import fnmatch
import json
import sys
from src.utils import cprint

# Style defaults
c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 

fs_m1 = 9  # for figure ticks
fs = 12  # for regular figure text
fs_p1 = 12  #  figure titles

golden_ratio = (5**.5 - 1) / 2
text_width = 5.50107
scale = 1
num_rows = 1
figsize = (6,5)
dpi = 300

matplotlib.rc('font', serif='Times')
    
matplotlib.rc('font', size=fs)          # controls default text sizes
matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=fs)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
matplotlib.rc('figure', titlesize=fs_p1)  # fontsize of the figure title


matplotlib.rc('font',**{'family':'serif','serif':['Times']})
matplotlib.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
matplotlib.rc('text', usetex=True)


matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

base_c10 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

base_c11k = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000']

title_dict = {"my_1d": "simple 1d", "matern_1d": "Matern", "wiggle": "wiggle", 
              "agw_1d": "Izmailov et al. (2019)", "andrew_1d": "Foong et al. (2019b)",
              "boston": "boston", "concrete": "concrete", "energy": "energy", 
              "naval": "naval", "power": "power", "protein": "protein", "yacht": "yacht",
              "wine": "wine", "kin8nm": "kin8nm"}

toy_list = ["wiggle", "matern_1d", "my_1d", "andrew_1d", "agw_1d"]
reg_list = ["boston", "concrete", "energy", "naval", "power", "protein", "yacht",
            "wine", "kin8nm"]

def plot_sidebyside_posterior_barplots(dir, n_runs, small, large, dataset):
    mkdir(f'{dir}/posterior_plots/')
    
    posteriors = []
    for i in range(n_runs):
        for file in os.listdir(f'{dir}/{i}/'):
            if fnmatch.fnmatch(file, f'{small}_*'):
                f_name = file
        try:
            with open(f'{dir}/{i}/{f_name}/media/query_results.json') as jsonfile:  
                data = json.load(jsonfile)
            posteriors.append(data['approx_d_posterior'])
        except FileNotFoundError:
            print(f'file not found: {dir}/{i}/{f_name}/media/query_results.json')
    posteriors = np.array(posteriors)
    means_small = posteriors.mean(axis=0)
    stds_small = posteriors.std(axis=0)

    posteriors = []
    for i in range(n_runs):
        for file in os.listdir(f'{dir}/{i}/'):
            if fnmatch.fnmatch(file, f'{large}_*'):
                f_name = file
        try:
            with open(f'{dir}/{i}/{f_name}/media/query_results.json') as jsonfile:
                    data = json.load(jsonfile)
            posteriors.append(data['approx_d_posterior'])                
        except FileNotFoundError:
            print(f'file not found: {dir}/{i}/{f_name}/media/query_results.json')
    posteriors = np.array(posteriors)
    means_large = posteriors.mean(axis=0)
    stds_large = posteriors.std(axis=0)
        
    x = np.array([i for i in range(means_small.shape[0])])
    width = 0.4 # width of bars

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.yaxis.grid(zorder=0,alpha=0.3)
    ax1 = ax.bar(x - width/2, means_small, width, yerr=stds_small, label=f'Train size: {small}', zorder=3)
    ax2 = ax.bar(x + width/2, means_large, width, yerr=stds_large, label=f'Train size: {large}', zorder=3)
    plt.title(title_dict[dataset])
    plt.xlabel(r'$d$')
    plt.ylabel(r'$q_{\phi}(d)$')
    plt.legend(loc='lower right')
    plt.savefig(f'{dir}/posterior_plots/{small}_{large}_d_post_approx.pdf', format='pdf', bbox_inches='tight')
    plt.close(fig=None)


def compare_methods(dataset, acq_fn, savedir, n_queries, init_train, query_size, measure='err', sgd=False):
    # Compare RMSE with error bars
    file_1 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}{acq_fn}_ntrain'
    file_2 = f'Dropout_{dataset}_3_100_0.001_0.0001_1_{init_train}{acq_fn}_ntrain'
    file_3 = f'MFVI_{dataset}_3_100_0.001_0.0001_1_{init_train}{acq_fn}_ntrain'
    if sgd:
        file_4 = f'SGD_{dataset}_3_100_0.001_0.0001_1_{init_train}{acq_fn}_ntrain'

    results_1 = np.genfromtxt(f'{savedir}/{file_1}/test_{measure}.csv', delimiter=',', dtype=None)
    results_2 = np.genfromtxt(f'{savedir}/{file_2}/test_{measure}.csv', delimiter=',', dtype=None)
    results_3 = np.genfromtxt(f'{savedir}/{file_3}/test_{measure}.csv', delimiter=',', dtype=None)
    if sgd:
        results_4 = np.genfromtxt(f'{savedir}/{file_4}/test_{measure}.csv', delimiter=',', dtype=None)
        
    means_1 = results_1.mean(axis=1)
    stds_1 = results_1.std(axis=1)
    means_2 = results_2.mean(axis=1)
    stds_2 = results_2.std(axis=1)
    means_3 = results_3.mean(axis=1)
    stds_3 = results_3.std(axis=1)
    if sgd:
        means_4 = results_4.mean(axis=1)
        stds_4 = results_4.std(axis=1)

    plt.figure(dpi=dpi, figsize=figsize)
    plt.grid(zorder=0,alpha=0.3)
    x = np.arange(init_train, init_train + n_queries*query_size, query_size)
    plt.plot(x, means_1, c=c[0], label='DUN', zorder=3)
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color=c[0], zorder=3)
    plt.plot(x, means_2, c=c[1], label='Dropout', zorder=3)
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color=c[1], zorder=3)
    plt.plot(x, means_3, c=c[2], label='MFVI', zorder=3)
    plt.fill_between(x, means_3+stds_3, means_3-stds_3, alpha=0.3, color=c[2], zorder=3)
    if sgd:
        plt.plot(x, means_4, c=c[3], label='SGD', zorder=3)
        plt.fill_between(x, means_4+stds_4, means_4-stds_4, alpha=0.3, color=c[3], zorder=3)
    plt.xlabel('Train set size')
    if measure=='err':
        plt.ylabel('Test RMSE')
    else:
        plt.ylabel('Test NLL')
    plt.title(title_dict[dataset])
    plt.legend()
    plt.savefig(f'./plots/{dataset}{acq_fn}_methods_{measure}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def compare_strategies(dataset, savedir, n_queries, init_train, query_size, measure='err', clip_var=True):
    # Compare RMSE with error bars
    file_1 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_ntrain'
    file_2 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_entropy_ntrain'
    if clip_var:
        file_3 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_variance_clip_ntrain'
    else:
        file_3 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_variance_ntrain'

    results_1 = np.genfromtxt(f'{savedir}/{file_1}/test_{measure}.csv', delimiter=',', dtype=None)
    results_2 = np.genfromtxt(f'{savedir}/{file_2}/test_{measure}.csv', delimiter=',', dtype=None)
    results_3 = np.genfromtxt(f'{savedir}/{file_3}/test_{measure}.csv', delimiter=',', dtype=None)
        
    means_1 = results_1.mean(axis=1)
    stds_1 = results_1.std(axis=1)
    means_2 = results_2.mean(axis=1)
    stds_2 = results_2.std(axis=1)
    means_3 = results_3.mean(axis=1)
    stds_3 = results_3.std(axis=1)

    plt.figure(dpi=dpi, figsize=figsize)
    plt.grid(zorder=0,alpha=0.3)
    x = np.arange(init_train, init_train + n_queries*query_size, query_size)
    plt.plot(x, means_1, c=c[6], label='Random', zorder=3)
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color=c[6], zorder=3)
    plt.plot(x, means_2, c=c[8], label='Max entropy', zorder=3)
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color=c[8], zorder=3)
    if clip_var:
        plt.plot(x, means_3, c=c[9], label='BALD (capped)', zorder=3)
    else:
        plt.plot(x, means_3, c=c[9], label='BALD', zorder=3)
    plt.fill_between(x, means_3+stds_3, means_3-stds_3, alpha=0.3, color=c[9], zorder=3)
    plt.xlabel('Train set size')
    if measure=='err':
        plt.ylabel('Test RMSE')
    else:
        plt.ylabel('Test NLL')
    plt.title(title_dict[dataset])
    plt.legend()
    if clip_var:
        plt.savefig(f'./plots/{dataset}_DUN_acq_funs_clipvar_{measure}.pdf', format='pdf', bbox_inches='tight')
    else:
        plt.savefig(f'./plots/{dataset}_DUN_acq_funs_{measure}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def compare_prior_decay(dataset, savedir, n_queries, init_train, query_size, measure='err'):
    # Compare RMSE with error bars
    file_1 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_variance_ntrain'
    file_2 = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_0.95_variance_ntrain'

    results_1 = np.genfromtxt(f'{savedir}/{file_1}/test_{measure}.csv', delimiter=',', dtype=None)
    results_2 = np.genfromtxt(f'{savedir}/{file_2}/test_{measure}.csv', delimiter=',', dtype=None)
        
    means_1 = results_1.mean(axis=1)
    stds_1 = results_1.std(axis=1)
    means_2 = results_2.mean(axis=1)
    stds_2 = results_2.std(axis=1)

    plt.figure(dpi=dpi, figsize=figsize)
    plt.grid(zorder=0,alpha=0.3)
    x = np.arange(init_train, init_train + n_queries*query_size, query_size)
    plt.plot(x, means_1, c=c[7], label='Uniform prior', zorder=3)
    plt.fill_between(x, means_1+stds_1, means_1-stds_1, alpha=0.3, color=c[7], zorder=3)
    plt.plot(x, means_2, c=c[3], label='Decaying prior', zorder=3)
    plt.fill_between(x, means_2+stds_2, means_2-stds_2, alpha=0.3, color=c[3], zorder=3)
    plt.xlabel('Train set size')
    if measure=='err':
        plt.ylabel('Test RMSE')
    else:
        plt.ylabel('Test NLL')
    plt.title(title_dict[dataset])
    plt.legend()
    plt.savefig(f'./plots/{dataset}_DUN_prior_decay_{measure}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


def generate_summary(dir, n_runs):
    test_err = []
    test_nll = []
    for i in range(n_runs):
        try:
            with open(f'{dir}/{i}/results_{i}.json') as jsonfile:
                data = json.load(jsonfile)
            test_err.append(data['test_err'])
            test_nll.append(data['test_nll'])
        except FileNotFoundError:
            print(f'Results for {dir}/{i}/ not found')
    test_err = np.array(test_err).T
    test_nll = np.array(test_nll).T
    # yacht var_clip remove outlies
    #test_err = np.delete(test_err, np.s_[18,5,21,35], axis=1)
    #test_nll = np.delete(test_nll, np.s_[18,5,21,35], axis=1)
    np.savetxt(f'{dir}/test_err.csv', test_err, delimiter=',')
    np.savetxt(f'{dir}/test_nll.csv', test_nll, delimiter=',')

if __name__=="__main__":
    n_runs = 40
    layers={"DUN":10, "Dropout":3, "MFVI":3, "SGD":3}
    init_train={"wiggle": 10, "agw_1d": 10, "andrew_1d": 10, "matern_1d": 10, "my_1d": 10, "boston": 20, "concrete": 50, "energy": 50, "kin8nm": 50, "naval": 50, "power": 50, "protein": 50, "wine": 50, "yacht": 20}
    query_size={"wiggle": 10, "agw_1d": 10, "andrew_1d": 5, "matern_1d": 10, "my_1d": 10, "boston": 20, "concrete": 20, "energy": 20, "kin8nm": 20, "naval": 20, "power": 20, "protein": 20, "wine": 20, "yacht": 10}
    n_queries={"wiggle": 20, "agw_1d": 30, "andrew_1d": 14, "matern_1d": 30, "my_1d": 30, "boston": 17, "concrete": 30, "energy": 30, "kin8nm": 30, "naval": 30, "power": 30, "protein": 30, "wine": 30, "yacht": 20}
    folder = 'saves_regression'

    ### Generate summary files
    '''Run once to collate results from all runs per experiment'''
    '''
    for method in ['DUN', 'Dropout', 'MFVI']:
        for acq_strategy in ['', '_entropy', '_variance', '_variance_clip']:
            for dataset in reg_list:
                    if dataset=='power':
                        n_runs = 38
                    if dataset=='protein':
                        n_runs = 30
                    dir = f"{folder}/{method}_{dataset}_{layers[method]}_100_0.001_0.0001_1_{init_train[dataset]}{acq_strategy}_ntrain"
                    generate_summary(dir, n_runs)
    '''
    ### Plotting
    # Posterior bar charts
    for dataset in reg_list:
        if dataset=='protein':
            n_runs=30
        for acq_strategy in ["_variance", "_variance_clip"]:
            dir = f'{folder}/DUN_{dataset}_10_100_0.001_0.0001_1_{init_train[dataset]}{acq_strategy}_ntrain'
            plot_sidebyside_posterior_barplots(dir, n_runs, 
                                            small=init_train[dataset], 
                                            large=init_train[dataset]+query_size[dataset]*(n_queries[dataset]-1),
                                            dataset=dataset
                                            )
    sys.exit('stop!')
    # Compare inference methods
    for dataset in reg_list:
        for acq_strategy in ["_variance", "_variance_clip"]:
            compare_methods(
                dataset, 
                acq_strategy, 
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='err', 
                sgd=False)
            compare_methods(dataset,
                acq_strategy, 
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='nll', 
                sgd=False)
    
    # Compare acq fns 
    for dataset in reg_list:
        for clip_var in [True, False]:
            compare_strategies(dataset, 
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='err', 
                clip_var=clip_var)
            compare_strategies(dataset,
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='nll', 
                clip_var=clip_var)
    
    # Compare prior decay
    for dataset in reg_list:
        compare_prior_decay(dataset, 
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='err')
        compare_prior_decay(dataset, 
                folder, 
                n_queries[dataset], 
                init_train[dataset], 
                query_size[dataset], 
                measure='nll')