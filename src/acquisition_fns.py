import numpy as np
import torch
from scipy import stats
from src.utils import cprint

from src.DUN.training_wrappers import DUN, DUN_VI

def acquire_samples(model, dataset, query_size=10, query_strategy='random', 
                    batch_size=128, num_workers=4):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    
    # create data loader for pool
    poolloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                             sampler=torch.utils.data.SubsetRandomSampler(unlabeled_idx))
       
    if query_strategy=='random':
        sample_idx = random_query(poolloader, query_size)
    elif query_strategy=='entropy':
        sample_idx = max_entropy_query(poolloader, model, query_size=query_size)
    elif query_strategy=='variance':
        sample_idx = max_pred_var_query(poolloader, model, query_size)
    else:
        raise Exception(f'{query_strategy} acquisition function not supported.')
    
    # update the labels for the selected indices
    for sample in sample_idx:
        dataset.update_label(sample)


def random_query(dataloader, query_size=10):
    """
    Randomly select samples from the pool for which to acquire labels.
    Since data is already shuffled in the DataLoader, simply select the first `query_size` samples.
    """
    torch.manual_seed(0)
    
    sample_idx = []
    
    for batch in dataloader:
        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break
    
    return sample_idx[0:query_size]


def max_entropy_query(dataloader, net, query_size=10, n_samples=1000, n_bins=10):
    '''Query points with max entropy. Entropy approximated via histogram of sampled predictions.'''    
    entropies = []
    indices = []
    
    d_posterior = net.prob_model.current_posterior.data.cpu().numpy()
    with torch.no_grad():
        for x, y, idx in dataloader:
            layer_preds = net.layer_predict(x).data.cpu().numpy()    
            for j in range(layer_preds.shape[1]):
                preds = layer_preds[:,j].reshape(-1)
                samples = np.random.choice(preds, size=n_samples, replace=True, p=d_posterior)
                hist, _ = np.histogram(samples, bins=n_bins)
                entropies.append(stats.entropy(hist))
                indices.append(idx[j])

    ent = np.asarray(entropies)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(ent)[::-1]
    
    return ind[sorted_pool][0:query_size]


def max_pred_var_query(dataloader, net, query_size=10):
    '''Query points with max model predictive variance (return_model_std=True).'''    
    stds = []
    indices = []
    
    with torch.no_grad():
        for x, y, idx in dataloader:
            if isinstance(net, (DUN, DUN_VI)):           
                pred_mu, pred_std = net.predict(x, get_std=True, return_model_std=True)
            else:
                pred_mu, pred_std = net.predict(x, Nsamples=100, return_model_std=True)        
            pred_std = pred_std.data.cpu().numpy()
            stds.extend(pred_std)
            indices.extend(idx)
    
    pred_stds = np.asarray(stds).reshape(1,-1)[0]
    # clip varaince at 1
    #pred_stds = np.where(pred_stds>1, 1, pred_stds)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(pred_stds)[::-1]

    return ind[sorted_pool][0:query_size]