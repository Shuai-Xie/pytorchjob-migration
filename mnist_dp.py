import os
import time

import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel

from migration import MigratableVariable
from models.resnet import *
from utils import *


def parse_args():
    parser = get_base_parser()
    args = parser.parse_args()
    # post process
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device_count = torch.cuda.device_count()
    args.batch_size = max(args.batch_size,
                          args.device_count * 2)  # min valid batchsize
    return args


def run(args):
    print('args:')
    pprint(vars(args))

    set_random_seeds(args.seed)

    # save dir
    mkdir(os.path.dirname(args.ckpt_path))

    # dataset
    train_loader, test_loader = get_dataloader(args)

    # model
    model = resnet18(num_classes=10).to(args.device)
    if args.device_count > 1:
        model = DataParallel(model)
        print('DP')

    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    # metircs to be recorded
    metircs = {'epoch': -1, 'best_epoch': -1, 'best_acc': 0.}

    start_epoch = 0
    if args.resume and os.path.isfile(args.ckpt_path):
        ckpt = load_ckpt(args.ckpt_path, model, optimizer)
        metircs = ckpt['metircs']
        print('load ckpt from epoch {}, metrics: {}'.format(
            metircs['epoch'], metircs))

    # make important vars migratable, which helps training models on k8s more secure.
    # migration starts
    model = MigratableVariable(model)
    optimizer = MigratableVariable(optimizer)
    metircs = MigratableVariable(metircs)
    # migration ends

    start_epoch = metircs['epoch'] + 1
    best_acc = metircs['best_acc']

    # train
    print('begin training')
    t1 = time.time()

    best_acc = 0.
    for epoch in range(start_epoch, args.epochs):
        train(args, model, train_loader, optimizer, criterion, epoch)
        test_acc = test(args, model, test_loader, epoch)
        metircs['epoch'] = epoch
        if test_acc >= best_acc:
            best_acc = test_acc
            print(f'test acc: {test_acc}, best acc: {best_acc}')
            # save ckpt
            metircs.update({'best_epoch': epoch, 'best_acc': best_acc})
            save_ckpt(args.ckpt_path, model, optimizer, metrics=metircs)
            print('save ckpt at epoch', epoch)

    t2 = time.time()
    print('training seconds:', t2 - t1)
    print('best_acc:', best_acc)


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
