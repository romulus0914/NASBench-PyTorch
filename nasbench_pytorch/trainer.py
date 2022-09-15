import numpy as np

import torch
from torch import nn


def train(net, train_loader, loss=None, optimizer=None, scheduler=None, grad_clip=5, num_epochs=10,
          num_validation=None, validation_loader=None, device=None, print_frequency=200,
          checkpoint_every_k=None, checkpoint_func=None):
    """
    Train a network from the NAS-bench-101 search space on a dataset (`train_loader`).

    Args:
        net: Network to train.
        train_loader: Train data loader.
        loss: Loss, default is CrossEntropyLoss.
        optimizer: Optimizer, default is SGD, possible: 'sgd', 'rmsprop', 'adam', or an optimizer object.
        scheduler: Default is CosineAnnealingLR.
        grad_clip: Gradient clipping parameter.
        num_epochs: Number of training epochs.
        num_validation: Number of validation examples (for print purposes).
        validation_loader: Optional validation set.
        device: Device to train on, default is cpu.
        print_frequency: How often to print info about batches.
        checkpoint_every_k: Every k epochs, save a checkpoint.
        checkpoint_func: Custom function to save the checkpoint, signature: func(net, metric_dict, epoch num)

    Returns:
        Final train (and validation) metrics.
    """

    net = net.to(device)

    # defaults
    if loss is None:
        loss = nn.CrossEntropyLoss()

    if optimizer is not None and not isinstance(optimizer, str):
        pass
    elif optimizer is None or optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4)
    elif optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(net.parameters(), lr=0.2, momentum=0.9, weight_decay=1e-4)
    elif optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters())

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # training

    n_batches = len(train_loader)

    metric_dict = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    for epoch in range(num_epochs):
        # checkpoint using a user defined function
        if checkpoint_every_k is not None and (epoch + 1) % checkpoint_every_k == 0:
            checkpoint_func(net, metric_dict, epoch + 1)

        net.train()

        train_loss = torch.tensor(0.0)
        correct = torch.tensor(0)
        total = 0

        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # forward
            outputs = net(inputs)

            # back-propagation
            optimizer.zero_grad()
            curr_loss = loss(outputs, targets)
            curr_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            # metrics
            train_loss += curr_loss.detach().cpu()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).sum().detach().cpu()

            if (batch_idx % print_frequency) == 0:
                print(f'Epoch={epoch}/{num_epochs} Batch={batch_idx + 1}/{n_batches} | '
                      f'Loss={train_loss / (batch_idx + 1):.3f}, '
                      f'Acc={correct / total:.3f}({correct}/{total})')

        last_loss = train_loss / (batch_idx + 1)
        acc = correct / total

        # save metrics
        metric_dict['train_loss'].append(last_loss.item())
        metric_dict['train_accuracy'].append(acc.item())

        if validation_loader is not None:
            test_metrics = test(net, validation_loader, loss, num_tests=num_validation, device=device)
            metric_dict['val_loss'].append(test_metrics['test_loss'])
            metric_dict['val_accuracy'].append(test_metrics['test_accuracy'])

        print('--------------------')
        scheduler.step()

    return metric_dict


def test(net, test_loader, loss=None, num_tests=None, device=None):
    """
    Evaluate the network on a test set.

    Args:
        net: Network for testing.
        test_loader: Test dataset.
        loss: Loss function, default is CrossEntropyLoss.
        num_tests: Number of test examples (for print purposes).
        device: Device to use.

    Returns:
        Test metrics.
    """
    net = net.to(device)
    net.eval()

    if loss is None:
        loss = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    n_tests = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)

            curr_loss = loss(outputs, targets)
            test_loss += curr_loss.detach()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).sum().detach()

            if num_tests is None:
                n_tests += len(targets)

        if num_tests is None:
            num_tests = n_tests

        print(f'Testing: Loss={(test_loss / len(test_loader)):.3f}, Acc={(correct / num_tests):.3f}'
              f'({correct}/{num_tests})')

    last_loss = test_loss / len(test_loader) if len(test_loader) > 0 else np.inf
    acc = correct / num_tests

    return {'test_loss': last_loss.item(), 'test_accuracy': acc.item()}
