import numpy as np
import torch

def acquire_samples(model, dataset, query_size=10, query_strategy='random', 
                    batch_size=128, num_workers=4):
    
    unlabeled_idx = np.nonzero(dataset.unlabeled_mask)[0]
    
    # create data loader for pool
    poolloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                             sampler=torch.utils.data.SubsetRandomSampler(unlabeled_idx))
       
    if query_strategy=='random':
        sample_idx = random_query(poolloader, query_size)
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
    sample_idx = []
    
    for batch in dataloader:
        _, _, idx = batch
        sample_idx.extend(idx.tolist())

        if len(sample_idx) >= query_size:
            break
    
    return sample_idx[0:query_size]