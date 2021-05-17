"""
Specific transforms and constants have been extracted from
  https://github.com/google-research/nasbench/blob/master/nasbench/lib/cifar.py
"""

import torch

import torchvision
import torchvision.transforms as transforms


def PrepareDataset(batch_size, test_batch_size=100):
    print('--- Preparing CIFAR10 Data ---')

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

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False)

    print('--- CIFAR10 Data Prepared ---')

    return trainloader, len(trainset), testloader, len(testset)
