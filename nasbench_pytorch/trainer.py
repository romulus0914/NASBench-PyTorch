import numpy as np

import torch
from torch import nn
from torch.autograd import Variable


def train(net, train_loader, loss=None, optimizer=None, scheduler=None, grad_clip=5, num_epochs=10,
          num_validation=None, validation_loader=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)

    # defaults
    if loss is None:
        loss = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.025, momentum=0.9, weight_decay=1e-4)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    n_batches = len(train_loader)

    for epoch in range(num_epochs):
        net.train()

        scheduler.step()

        train_loss = 0
        correct = 0
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

            train_loss += curr_loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).sum().item()

            # TODO print only sometimes, avg train loss only over some 2k batches or so
            print(f'Epoch={epoch}/{num_epochs} Batch={batch_idx + 1}/{n_batches} | '
                  f'Loss={train_loss/(batch_idx+1):.3f}, '
                  f'Acc={correct/total:.3f}({correct}/{total})')

        last_loss = train_loss / (batch_idx + 1) if batch_idx > 0 else np.inf
        acc = correct / total

        if validation_loader is not None:
            val_loss, val_acc = test(net, validation_loader, loss, num_tests=num_validation, device=device)
            return last_loss, acc, val_loss, val_acc

        return last_loss, acc


def test(net, test_loader, loss=None, num_tests=None, device=None):
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
            test_loss += curr_loss.item()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).sum().item()

            if num_tests is None:
                n_tests += len(targets)

        if num_tests is None:
            num_tests = n_tests

        print(f'Testing: Loss={(test_loss / len(test_loader)):.3f}, Acc={(correct / num_tests):.3f}'
              f'({correct}/{num_tests})')

    last_loss = test_loss / len(test_loader) if len(test_loader) > 0 else np.inf
    acc = correct / num_tests

    return last_loss, acc
