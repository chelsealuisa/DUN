import numpy as np
from numpy.lib.arraysetops import isin
from scipy.special import softmax
import random
import sys
import torch
from torch.nn import functional as F
from scipy import stats
from src.utils import cprint
from src.DUN.training_wrappers import DUN, DUN_VI


def acquire_samples(model, dataset, query_size=10, query_strategy='random', batch_size=2048, 
                    num_workers=4, clip_var=False, bias_reduction_weights=False, seed=0):
    
    torch.manual_seed(seed)
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    
    # create data loader for pool
    poolloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                             sampler=torch.utils.data.SubsetRandomSampler(unlabeled_idx))
       
    if query_strategy=='random':
        sample_idx = random_query(poolloader, query_size, dataset.N, seed=seed)
        for sample in sample_idx:
            dataset.update_label(sample)
        return
    elif query_strategy=='entropy':
        sample_idx, q_scores = max_entropy_query(poolloader, model, query_size, bias_weights=bias_reduction_weights)
    elif query_strategy=='variance':
        sample_idx, q_scores = max_pred_var_query(poolloader, model, query_size, clip_var, bias_weights=bias_reduction_weights)
    else:
        raise Exception(f'{query_strategy} acquisition function not supported.')
    
    # update the labels for the selected indices
    for sample, q_score in zip(sample_idx, q_scores):
        dataset.update_label(sample, q_score)


def random_query(dataloader, query_size, N, seed=0):
    """
    Randomly select samples from the pool for which to acquire labels.
    Since data is already shuffled in the DataLoader, simply select the first `query_size` samples.
    """
    torch.manual_seed(seed)
    
    sample_idx = []
    for batch in dataloader:
        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break
    
    return sample_idx[0:query_size]


def max_entropy_query(dataloader, net, query_size=10, n_samples=1000, n_bins=10, bias_weights=False, T=1):
    '''Query points with max entropy. For regression, entropy approximated via histogram of sampled predictions.'''    
    entropies = []
    indices = []
    
    if net.regression:
        if isinstance(net, (DUN, DUN_VI)): 
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
        else:
            with torch.no_grad():
                samples = []
                hists = []
                for x, _, idx in dataloader:
                    indices.extend(idx)
                    batch_samples = []
                    for _ in range(n_samples):
                        x1 = net.model.layers(x)
                        batch_samples.append(x1.data)
                    batch_samples = torch.stack(batch_samples, dim=0).squeeze(2)
                    samples.append(batch_samples)
                samples = torch.cat(samples, dim=1)
                for j in range(samples.shape[1]):
                    hist, _ = np.histogram(samples[:,j], bins=n_bins)
                    hists.append(hist)
                hists = np.array(hists)
                entropies.extend(stats.entropy(hists, axis=1))
    else:
        with torch.no_grad():
            for x, y, idx in dataloader:
                if isinstance(net, (DUN, DUN_VI)): 
                    preds = net.predict(x).data.cpu().numpy()
                else:
                    preds = net.predict(x, Nsamples=100, return_model_std=False)
                entropies.extend(stats.entropy(preds, axis=1))
                indices.extend(idx)

    ent = np.asarray(entropies)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(ent)[::-1]
    ent = T*ent # tempering
    softmax_scores = softmax(ent)

    if bias_weights:
        indices = []
        q_scores = []
        for i in range(query_size):
            # sample a point with probability = softmax score
            choice = random.choices(ind, weights=softmax_scores, k=1)[0]
            q_scores.append(softmax_scores[np.where(ind==choice)][0])
            # remove is from the pool and recompute softmaxes
            indices.append(choice)
            ent = np.delete(ent, np.where(ind==choice))
            ind = np.delete(ind, np.where(ind==choice))
            softmax_scores = softmax(ent)
        return indices, q_scores
    else:
        return ind[sorted_pool][0:query_size], softmax_scores[sorted_pool][0:query_size]


def max_pred_var_query(dataloader, net, query_size=10, clip_var=False, n_samples=50, bias_weights=False, T=1):
    '''
    Query points with max model predictive variance (return_model_std=True).
    Equivalent to BALD acquisition.
    '''    
    stds = []
    indices = []
    
    with torch.no_grad():
        if net.regression:
            for x, y, idx in dataloader:
                if isinstance(net, (DUN, DUN_VI)):           
                    _, pred_std = net.predict(x, get_std=True, return_model_std=True)
                else:
                    _, pred_std = net.predict(x, Nsamples=100, return_model_std=True)        
                pred_std = pred_std.data.cpu().numpy()
                stds.extend(pred_std)
                indices.extend(idx)
        else:
            if isinstance(net, (DUN, DUN_VI)):
                d_posterior = net.prob_model.current_posterior.data.cpu().numpy()
                for x, y, idx in dataloader:
                    marg_preds = net.predict(x, get_std=False, return_model_std=False).data.cpu().numpy() # shape = (pool size, no. classes)
                    layer_preds = net.layer_predict(x).data.cpu().numpy() # shape = (depth, pool size, no. classes)
                    # BALD approx
                    model_var = stats.entropy(marg_preds, axis=1) # 1st term in BALD
                    per_d_entropy = stats.entropy(layer_preds, axis=2)
                    noise = (d_posterior*np.transpose(per_d_entropy)).sum(axis=1) # 2nd term in BALD
                    bald = model_var - noise
                    stds.extend(bald)
                    indices.extend(idx)
            else:
                with torch.no_grad():
                    samples = []
                    for x, y, idx in dataloader:
                        indices.extend(idx)
                        batch_samples = []
                        for _ in range(n_samples):
                            x1 = net.model.layers(x)
                            batch_samples.append(x1.data)
                        batch_samples = torch.stack(batch_samples, dim=0)
                        samples.append(batch_samples)
                    samples = torch.cat(samples, dim=1)
                    # BALD approx
                    probs = F.softmax(samples, dim=2)
                    mean_probs = torch.sum(probs, dim=0) / probs.shape[0]
                    model_var = stats.entropy(mean_probs, axis=1) # 1st term in BALD
                    noise = np.zeros(probs.shape[1])
                    for i in range(probs.shape[1]):
                        probs_np = probs[:,i,:].numpy()
                        noise[i] = -np.sum(probs_np*np.log(probs_np))
                    noise = noise / probs.shape[0] # 2nd term in BALD
                    bald = model_var - noise
                    stds.extend(bald)
                    
    pred_stds = np.asarray(stds).reshape(1,-1)[0]
    if clip_var:
        # clip the variances at 1
        pred_stds = np.where(pred_stds>1, 1, pred_stds)
    ind = np.asarray(indices)
    sorted_pool = np.argsort(pred_stds)[::-1]
    pred_stds = T*pred_stds # tempering
    softmax_scores = softmax(pred_stds)

    if bias_weights:
        indices = []
        q_scores = []
        for i in range(query_size):
            # sample a point with probability = softmax score
            choice = random.choices(ind, weights=softmax_scores, k=1)[0]
            q_scores.append(softmax_scores[np.where(ind==choice)][0])
            # remove is from the pool and recompute softmaxes
            indices.append(choice)
            pred_stds = np.delete(pred_stds, np.where(ind==choice))
            ind = np.delete(ind, np.where(ind==choice))
            softmax_scores = softmax(pred_stds)
        return indices, q_scores
    else:
        return ind[sorted_pool][0:query_size], softmax_scores[sorted_pool][0:query_size]