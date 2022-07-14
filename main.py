import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from nasbench_pytorch.datasets.cifar10 import prepare_dataset
from nasbench_pytorch.model import Network
from nasbench_pytorch.model import ModelSpec
from nasbench_pytorch.trainer import train, test

matrix = [[0, 1, 1, 1, 0, 1, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 1],
          [0, 0, 0, 0, 0, 0, 0]]

operations = ['input', 'conv1x1-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'conv3x3-bn-relu',
              'maxpool3x3', 'output']


def save_checkpoint(net, postfix='cifar10'):
    print('--- Saving Checkpoint ---')

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')

    torch.save(net.state_dict(), './checkpoint/ckpt_' + postfix + '.pt')

def reload_checkpoint(path, device=None):
    print('--- Reloading Checkpoint ---')

    assert os.path.isdir('checkpoint'), '[Error] No checkpoint directory found!'
    return torch.load(path, map_location=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NASBench')
    parser.add_argument('--random_state', default=1, type=int, help='Random seed.')
    parser.add_argument('--data_root', default='./data/', type=str, help='Path where cifar will be downloaded.')
    parser.add_argument('--module_vertices', default=7, type=int, help='#vertices in graph')
    parser.add_argument('--max_edges', default=9, type=int, help='max edges in graph')
    parser.add_argument('--available_ops', default=['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'],
                        type=list, help='available operations performed on vertex')
    parser.add_argument('--in_channels', default=3, type=int, help='Number of input channels.')
    parser.add_argument('--stem_out_channels', default=128, type=int, help='output channels of stem convolution')
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='#modules per stack')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--test_batch_size', default=100, type=int, help='test set batch size')
    parser.add_argument('--epochs', default=100, type=int, help='#epochs of training')
    parser.add_argument('--validation_size', default=0, type=int, help="Size of the validation set to split off.")
    parser.add_argument('--num_workers', default=0, type=int, help="Number of parallel workers for the train dataset.")
    parser.add_argument('--learning_rate', default=0.025, type=float, help='base learning rate')
    parser.add_argument('--lr_decay_method', default='COSINE_BY_STEP', type=str, help='learning decay method')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 regularization weight')   
    parser.add_argument('--grad_clip', default=5, type=float, help='gradient clipping')
    parser.add_argument('--load_checkpoint', default='', type=str, help='Reload model from checkpoint')
    parser.add_argument('--num_labels', default=10, type=int, help='#classes')
    parser.add_argument('--device', default='cuda', type=str, help='Device for network training.')
    parser.add_argument('--print_freq', default=100, type=int, help='Batch print frequency.')

    args = parser.parse_args()

    # cifar10 dataset
    dataset = prepare_dataset(args.batch_size, test_batch_size=args.test_batch_size, root=args.data_root,
                              validation_size=args.validation_size, random_state=args.random_state,
                              set_global_seed=True, num_workers=args.num_workers)

    train_loader, test_loader, test_size = dataset['train'], dataset['test'], dataset['test_size']
    valid_loader = dataset['validation'] if args.validation_size > 0 else None

    # model
    spec = ModelSpec(matrix, operations)
    net = Network(spec, num_labels=args.num_labels, in_channels=args.in_channels, stem_out_channels=args.stem_out_channels,
                  num_stacks=args.num_stacks, num_modules_per_stack=args.num_modules_per_stack)

    if args.load_checkpoint != '':
        net.load_state_dict(reload_checkpoint(args.load_checkpoint))
    net.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    result = train(net, train_loader, loss=criterion, optimizer=optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
                   num_epochs=args.epochs, num_validation=args.validation_size, validation_loader=valid_loader,
                   device=args.device, print_frequency=args.print_freq)

    last_epoch = {k: v[-1] for k, v in result.items() if len(v) > 0}
    print(f"Final train metrics: {last_epoch}")

    result = test(net, test_loader, loss=criterion, num_tests=test_size, device=args.device)
    print(f"\nFinal test metrics: {result}")

    save_checkpoint(net)
