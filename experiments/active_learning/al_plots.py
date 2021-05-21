import numpy as np
from src.plots import plot_al_mean_rmse, plot_al_rmse
'''
# Mean and std per method
files = [
    'Dropout_agw_1d_3_100_0.001_0.0001_1',
    'DUN_agw_1d_10_100_0.001_0.0001_1',
    'MFVI_agw_1d_3_100_0.001_0.0001_1',
    'SGD_agw_1d_3_100_0.001_0.0001_1',
    'Dropout_wiggle_3_100_0.001_0.0001_1',
    'DUN_wiggle_10_100_0.001_0.0001_1',
    'MFVI_wiggle_3_100_0.001_0.0001_1',
    'SGD_wiggle_3_100_0.001_0.0001_1',
]

for file in files:
    results = np.genfromtxt(f'saves/{file}/results.csv', delimiter=',')
    means = results[:,0]
    stds = results[:,1]
    if file.split('_')[1]=='wiggle':
        n_queries = 20
    else:
        n_queries = 30
    query_size = 10
    plot_al_rmse(f'saves/{file}/rmse_plot', means, stds, n_queries, query_size)
'''

# Mean of all methods
for dataset in ['agw_1d', 'wiggle']:
    if dataset=='wiggle':
        n_queries = 20
    else:
        n_queries = 30
    query_size = 10
    dun = np.genfromtxt(f'saves/DUN_{dataset}_10_100_0.001_0.0001_1/results.csv', delimiter=',')
    dropout = np.genfromtxt(f'saves/Dropout_{dataset}_3_100_0.001_0.0001_1/results.csv', delimiter=',')
    mfvi = np.genfromtxt(f'saves/MFVI_{dataset}_3_100_0.001_0.0001_1/results.csv', delimiter=',')
    sgd = np.genfromtxt(f'saves/SGD_{dataset}_3_100_0.001_0.0001_1/results.csv', delimiter=',')

    plot_al_mean_rmse(f'saves/{dataset}_rmse_all', dun, dropout, mfvi, sgd, dataset, n_queries, query_size)
