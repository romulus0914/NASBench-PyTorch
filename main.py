import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec

import os
import argparse

matrix = [[0, 1, 1, 1, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0]]
operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']

def PrepareDataset(batch_size):
    print('--- Preparing CIFAR10 Data ---')

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # source https://github.com/google-research/nasbench/blob/master/nasbench/lib/cifar.py
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

    print('--- CIFAR10 Data Prepared ---')

    return trainloader, len(trainset), testloader, len(testset)

def _ToModelSpec(mat, ops):
    return ModelSpec(mat, ops)

def Train(net, trainloader, testloader, criterion, optimizer, scheduler, num_trains, num_tests, args):
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate

    for epoch in range(num_epochs):
        net.train()

        scheduler.step()

        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

            # forward
            outputs = net(inputs)

            # back-propagation
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predict.eq(targets.data).cpu().sum().item()

            print('Epoch=%d Batch=%d | Loss=%.3f, Acc=%.3f(%d/%d)' %
                  (epoch, batch_idx+1, train_loss/(batch_idx+1), correct/total, correct, total))

        # testing
        Test(net, testloader, criterion, num_tests)

def Test(net, testloader, criterion, num_tests, predict_net=None):
    net.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = Variable(inputs.cuda()), Variable(targets.cuda())

            outputs = net(inputs)

            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predict = torch.max(outputs.data, 1)
            correct += predict.eq(targets.data).cpu().sum().item()

        print('Testing: Loss=%.3f, Acc=%.3f(%d/%d)' %
              (test_loss/len(testloader), correct/num_tests, correct, num_tests))

def SaveCheckpoint(net, postfix='cifar10'):
    print('--- Saving Checkpoint ---')

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(net.state_dict(), './checkpoint/ckpt.'+postfix)

def ReloadCheckpoint(path):
    print('--- Reloading Checkpoint ---')

    assert os.path.isdir('checkpoint'), '[Error] No checkpoint directory found!'
    return torch.load(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NASBench')
    parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
    parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
    parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                        type=list, help='available operations performed on vertex')
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules') 
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
    parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
    parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')   
    parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
    parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
    parser.add_argument('--num_labels', default=10, type=int, help='#classes')

    args = parser.parse_args()

    # cifar10 dataset
    trainloader, num_trains, testloader, num_tests = PrepareDataset(args.batch_size)

    # model
    spec = _ToModelSpec(matrix, operations)
    net = Network(spec, args)
    if args.load_checkpoint != '':
        net.load_state_dict(ReloadCheckpoint(args.load_checkpoint))
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    Train(net, trainloader, testloader, criterion, optimizer, scheduler, num_trains, num_tests, args)
    SaveCheckpoint(net)
