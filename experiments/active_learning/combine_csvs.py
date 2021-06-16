import numpy as np

dir = 'saves/DUN_wiggle_10_100_0.001_0.0001_1_50' #DUN_concrete_10_100_0.001_0.0001_1_50__v3

res_list = []
for i in range(5):
    res_list.append(np.genfromtxt(f'{dir}/{i}/results_{i}.csv', delimiter=','))

results = np.zeros((res_list[0].shape[0], 5))
for i in range(5):
    results[:,i] = res_list[i]

means = results.mean(axis=1).reshape(-1,1)
stds = results.std(axis = 1).reshape(-1,1)
results = np.concatenate((means, stds, results), axis=1)
np.savetxt(f'{dir}/results.csv', results, delimiter=',')