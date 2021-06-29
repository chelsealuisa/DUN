import numpy as np
from src.plots import plot_al_mean_rmse, plot_al_rmse
import matplotlib.pyplot as plt

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
'''
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

# Mean of all methods
for dataset in ['wiggle']: #['agw_1d', 'wiggle']:
    if dataset=='wiggle':
        n_queries = 20
    else:
        n_queries = 30
    query_size = 10
    init_train = 10
    dun = np.genfromtxt(f'{savedir}/DUN_{dataset}_10_100_0.001_0.0001_1_{init_train}_variance/results.csv', delimiter=',')[:,0]
    dropout = np.genfromtxt(f'{savedir}/Dropout_{dataset}_3_100_0.001_0.0001_1_{init_train}_variance_p_drop_0.05_5000ep/results.csv', delimiter=',')[:,0]
    mfvi = np.genfromtxt(f'{savedir}/MFVI_{dataset}_3_100_0.001_0.0001_1_{init_train}_variance_5000ep/results.csv', delimiter=',')[:,0]
    #sgd = np.genfromtxt(f'{savedir}/SGD_{dataset}_3_100_0.001_0.0001_1_{init_train}_variance/results.csv', delimiter=',')[:,0]
    #plot_al_mean_rmse(f'{savedir}/{dataset}_rmse_all_variance_5000ep', dun, dropout, mfvi, sgd, dataset, n_queries, query_size, init_train)
    plot_al_mean_rmse(f'{savedir}/{dataset}_rmse_all_variance_5000ep', dun, dropout, mfvi, dataset, n_queries, query_size, init_train)

'''

# Compare acquisition functions
dataset = 'wiggle'
savedir = 'saves'
n_queries = 20
init_train_size = 10
query_size = 10

file_ran = f'DUN_{dataset}_10_20_0.001_0.0001_1_{init_train_size}_variance_clip_ntrain'
file_ent = f'DUN_{dataset}_10_20_0.001_0.0001_1_{init_train_size}_0.95_variance_clip_ntrain'
#file_var = f'DUN_{dataset}_10_100_0.001_0.0001_1_{init_train_size}_0.95_variance_ntrain'
results_ran = np.genfromtxt(f'{savedir}/{file_ran}/results.csv', delimiter=',')
results_ent = np.genfromtxt(f'{savedir}/{file_ent}/results.csv', delimiter=',')
#results_var = np.genfromtxt(f'{savedir}/{file_var}/results.csv', delimiter=',')
means_ran = results_ran[:,0]
stds_ran = results_ran[:,1]
means_ent = results_ent[:,0]
stds_ent = results_ent[:,1]
#means_var = results_var[:,0]
#stds_var = results_var[:,1]

plt.figure(dpi=300)
x = np.arange(init_train_size, init_train_size + n_queries*query_size, query_size)
plt.plot(x, means_ran, c='tab:blue', label='max variance clipped, uniform prior')
plt.fill_between(x, means_ran+stds_ran, means_ran-stds_ran, alpha=0.3, color='tab:blue')
plt.plot(x, means_ent, c='tab:red', label='max variance clipped, 0.95 prior decay')
plt.fill_between(x, means_ent+stds_ent, means_ent-stds_ent, alpha=0.3, color='tab:red')
#plt.plot(x, means_var, c='tab:green', label='max variance, 0.95 prior decay')
#plt.fill_between(x, means_var+stds_var, means_var-stds_var, alpha=0.3, color='tab:green')
plt.xlabel('Train set size')
plt.ylabel('Validation RMSE')
plt.title(f'{dataset} dataset')
plt.legend()
plt.savefig( f'{savedir}/{dataset}_var_clipped_prior_decay_0.95.pdf', format='pdf', bbox_inches='tight')
plt.close()