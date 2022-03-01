"""
Specific transforms and constants have been extracted from
  https://github.com/google-research/nasbench/blob/master/nasbench/lib/cifar.py
"""
import random

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


def prepare_dataset(batch_size, test_batch_size=100, root='./data/', validation_size=0, random_state=None,
                    no_valid_transform=False, num_workers=0):
    print('\n--- Preparing CIFAR10 Data ---')

    if random_state is not None:
        g = torch.Generator()
        g.manual_seed(random_state)

    def worker_init_fn(worker_id):
        seed = random_state + worker_id
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

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
                                                   worker_init_fn=worker_init_fn)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   sampler=valid_sampler, num_workers=num_workers,
                                                   worker_init_fn=worker_init_fn)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                   worker_init_fn=worker_init_fn)
        valid_loader = None

    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False,
                                              num_workers=num_workers, worker_init_fn=worker_init_fn)
    test_size = len(test_set)

    print('--- CIFAR10 Data Prepared ---\n')

    if validation_size > 0:
        return train_loader, train_size - validation_size, valid_loader, validation_size, test_loader, test_size

    return train_loader, train_size, test_loader, test_size
