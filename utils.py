import argparse
import os
import random
import shutil
import time
from pprint import pprint

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def set_random_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # most important
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.enabled = False  # 禁用 cudnn 使用非确定性算法
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:  # faster, less reproducible
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def print_random_vals(rank, num=1):
    for n in range(num):
        print('rank {} group {}, random: {},  np: {}, torch: {}'.format(
            rank, n, random.random(), np.random.rand(1), torch.rand(1)))


def curtime():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def mkdir(path, exist_ok=True):
    if os.path.exists(path) and not exist_ok:
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=exist_ok)


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt-path', default='ckpts/best.pth', type=str)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    return parser


def get_dataloader(args):
    trans = transforms.Compose([
        # transforms.Resize(224),  # mnist 28 -> 224
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    train_set = datasets.FashionMNIST('data',
                                      train=True,
                                      download=False,
                                      transform=trans)
    test_set = datasets.FashionMNIST('data',
                                     train=False,
                                     download=False,
                                     transform=trans)

    kwargs = {'pin_memory': False, 'num_workers': 4}

    # ddp, DistributedSampler default get rank from process group
    if hasattr(args, 'world_size'):
        train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
        train_loader = DataLoader(train_set,
                                  args.batch_size,
                                  sampler=train_sampler,
                                  **kwargs)
    else:
        train_loader = DataLoader(train_set,
                                  args.batch_size,
                                  shuffle=True,
                                  **kwargs)

    test_loader = DataLoader(test_set, args.batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader


# todo maybe a decorator to find when to save the model
def train(args, model, train_loader, optimizer, criterion, epoch, vis=True):
    model.train()
    num_batches = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if vis and (batch_idx % args.log_interval == 0
                    or batch_idx + 1 == num_batches):
            print('Train Epoch: {} [{}/{}]\tloss={:.4f}'.format(
                epoch, batch_idx, len(train_loader), loss.item()))


@torch.no_grad()
def test(args, model, test_loader, epoch, vis=True):
    model.eval()
    correct = total = 0
    num_batches = len(test_loader)
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        if vis and (batch_idx % args.log_interval == 0
                    or batch_idx + 1 == num_batches):
            test_acc = 100. * correct / total
            print('Test Epoch: {} [{}/{}]\tacc={:.4f}'.format(
                epoch, batch_idx, len(test_loader), test_acc))

    test_acc = 100. * correct / total
    print('Test Epoch: {}, acc={:.4f}'.format(epoch, test_acc))
    return test_acc


# we suggest store the migrated values in dict to help migration
def save_ckpt(path, model, optimizer, metrics: dict = None):
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(ckpt, path, _use_new_zipfile_serialization=False)


def load_ckpt(path, model, optimizer, map_location='cpu'):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optim_state_dict'])
    return ckpt
