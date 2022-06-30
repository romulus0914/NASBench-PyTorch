"""
Specific transforms and constants have been extracted from
  https://github.com/google-research/nasbench/blob/master/nasbench/lib/cifar.py
"""
import random
from functools import partial
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler


def train_valid_split(dataset_size, valid_size, random_state=None):
    random = np.random.RandomState(seed=random_state) if random_state is not None else np.random
    valid_inds = random.choice(dataset_size, size=valid_size, replace=False)

    train_inds = np.delete(np.arange(dataset_size), valid_inds)

    return SubsetRandomSampler(train_inds), SubsetRandomSampler(valid_inds)


def seed_worker(seed, worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_dataset(batch_size, test_batch_size=100, root='./data/', validation_size=0, random_state=None,
                    set_global_seed=False, no_valid_transform=False,
                    num_workers=0, num_val_workers=0, num_test_workers=0):
    """
    Download the CIFAR-10 dataset and prepare train and test DataLoaders (optionally also validation loader).

    Args:
        batch_size: Batch size for the train (and validation) loader.
        test_batch_size: Batch size for the test loader.
        root: Directory path to download the CIFAR-10 dataset to.
        validation_size: Size of the validation dataset to split off the train set.
            If  == 0, don't return the validation set.

        random_state: Seed for the random functions (generators from numpy and random)
        set_global_seed: If True, call np.random.seed(random_state) and random.seed(random_state). Useful when
            using 0 workers (because otherwise RandomCrop will return different results every call), but affects
            the seed in the whole program.

        no_valid_transform: If True, don't use RandomCrop and RandomFlip for the validation set.
        num_workers: Number of workers for the train loader.
        num_val_workers: Number of workers for the validation loader.
        num_test_workers: Number of workers for the test loader.

    Returns:
        if validation_size > 0:
            train loader, train size, validation loader, validation size, test loader, test size
        otherwise:
            train loader, train size, test loader, test size

        The sizes are dataset sizes, not the number of batches.

    """

    if set_global_seed:
        seed_worker(random_state, 0)

    if random_state is not None:
        worker_fn = partial(seed_worker, random_state)
    else:
        worker_fn=None

    print('\n--- Preparing CIFAR10 Data ---')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transform)
    valid_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=test_transform)
    valid_set = valid_set if no_valid_transform else train_set
    train_size = len(train_set)

    # split off random validation set
    if validation_size > 0:
        train_sampler, valid_sampler = train_valid_split(train_size, validation_size, random_state=random_state)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False,
                                                   sampler=train_sampler, num_workers=num_workers,
                                                   worker_init_fn=worker_fn)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   sampler=valid_sampler, num_workers=num_val_workers,
                                                   worker_init_fn=worker_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=worker_fn)
        valid_loader = None

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_test_workers, worker_init_fn=worker_fn)
    test_size = len(test_set)

    print('--- CIFAR10 Data Prepared ---\n')

    data = {
        'train': train_loader,
        'train_size': train_size,
        'test': test_loader,
        'test_size': test_size
    }

    if validation_size > 0:
        data['train_size'] = train_size - validation_size
        data['validation'] = valid_loader
        data['validation_size'] = validation_size

    return data
