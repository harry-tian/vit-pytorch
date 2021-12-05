# -*- coding: utf-8 -*-

from __future__ import print_function

import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import os
import argparse
import csv
import time
import glob
from itertools import chain
import random
import zipfile
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch import ViT
from ipywidgets import FloatProgress
from utils import progress_bar
import wandb

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
# if args.aug:  
#     N = 2; M = 14;
#     transform_train.transforms.insert(0, RandAugment(N, M))trainset  = datasets.CIFAR10(root=train_dir, train=True, transform=transform_train, download=True)

def train(epoch, model, trainloader, criterion, optimizer, device, use_amp):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

def test(epoch, model, testloader, criterion, optimizer, device, use_amp, best_acc, patch):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    # if not args.cos:
    #     scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": model.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vit-{}-ckpt.t7'.format(patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_vit_patch{patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc, best_acc
    
def main():
    seed_everything(42)
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=3e-5, type=float, help='learning rate') 
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--aug', action='store_true', help='use randomaug')
    parser.add_argument('--amp', action='store_true', help='enable AMP training')
    parser.add_argument('--model', default='vit')
    parser.add_argument('--batch_size', default='256')
    parser.add_argument('--n_epochs', type=int, default='50')
    parser.add_argument('--patch', default='4', type=int)
    parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
    parser.add_argument("--wandb_run_name", default=None, type=str)
    args = parser.parse_args()

    wandb.init(project="cifar10-challange")
    wandb.run.name = args.wandb_run_name
    wandb.config.update(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    print('==> Preparing data..')

    trainset  = datasets.CIFAR10(root='data/cifar10/train', train=True, transform=transform_train, download=True)
    testset  = datasets.CIFAR10(root='data/cifar10/test', train=False, transform=transform_test, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size), shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=int(args.batch_size), shuffle=True, num_workers=8)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('==> Building model..')
    model = ViT(
        image_size = 32,
        patch_size = args.patch,
        num_classes = 10,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
        )

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
        cudnn.benchmark = True

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.model))
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr) if args.opt == "adam" else optim.SGD(model.parameters(), lr=args.lr)  

    if not args.cos:
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

    if args.cos:
        wandb.config.scheduler = "cosine"
    else:
        wandb.config.scheduler = "ReduceLROnPlateau"

    list_loss = []
    list_acc = []

    wandb.watch(model)
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainloss = train(epoch, model, trainloader, criterion, optimizer, device, args.amp)
        val_loss, acc, best_acc = test(epoch, model, testloader, criterion, optimizer, device, args.amp, best_acc, args.patch)
        
        if args.cos:
            scheduler.step(epoch-1)
        
        list_loss.append(val_loss)
        list_acc.append(acc)
        
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": time.time()-start})

        with open(f'log/log_{args.model}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss) 
            writer.writerow(list_acc) 
        print(list_loss)

    wandb.save("wandb_{}.h5".format(args.model))

main()