import torch
from torch import nn
from torch.autograd import Variable


def train(net, train_loader, loss, optimizer, scheduler, grad_clip, num_epochs=10,
          num_validation=None, validation_loader=None):

    n_batches = len(train_loader)

    for epoch in range(num_epochs):
        net.train()

        scheduler.step()

        train_loss = 0
        correct = 0
        total = 0

        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

            # forward
            outputs = net(inputs)

            # back-propagation
            optimizer.zero_grad()
            loss = loss(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum().item()

            print(f'Epoch={epoch}/{num_epochs} Batch={batch_idx + 1}/{n_batches} | '
                  f'Loss={train_loss/(batch_idx+1): %.3f}, '
                  f'Acc={correct/total: %.3f}({correct}/{total})')

        if validation_loader is not None:
            test(net, validation_loader, loss, num_tests=num_validation)


def test(net, test_loader, criterion, num_tests=None):
    net.eval()

    test_loss = 0
    correct = 0

    n_tests = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).cpu().sum().item()

            if num_tests is None:
                n_tests += len(targets)

        if num_tests is None:
            num_tests = n_tests

        print('Testing: Loss=%.3f, Acc=%.3f(%d/%d)' %
              (test_loss / len(test_loader), correct / num_tests, correct, num_tests))
