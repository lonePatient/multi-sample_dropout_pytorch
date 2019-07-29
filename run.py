import torch
import argparse
import torch.nn as nn
from nn import MiniResNet
from tools import AverageMeter
from progressbar import ProgressBar
from tools import seed_everything
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from trainingmonitor import TrainingMonitor

epochs = 20
batch_size = 128
seed = 42
arch = 'CNNNet2'
learning_rate = 0.01
device = torch.device("cuda:0")
seed_everything(seed)

def train(train_loader):
    pbar = ProgressBar(n_batch=len(train_loader))
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    count = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, loss = model(data,y = target,loss_fn = nn.CrossEntropyLoss())
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        count += data.size(0)
        train_acc.update(correct, n=1)
        pbar.batch_step(batch_idx=batch_idx, info={'loss': loss.item(), 'acc': correct / data.size(0)},
                        bar_type='Training')
        train_loss.update(loss.item(), n=1)
    print(' ')
    return {'loss': train_loss.avg,
            'acc': train_acc.sum / count}

def test(test_loader):
    pbar = ProgressBar(n_batch=len(test_loader))
    valid_loss = AverageMeter()
    valid_acc = AverageMeter()
    model.eval()
    count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output, loss = model(data,y = target,loss_fn = nn.CrossEntropyLoss())
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            valid_loss.update(loss, n=data.size(0))
            valid_acc.update(correct, n=1)
            count += data.size(0)
            pbar.batch_step(batch_idx=batch_idx, info={}, bar_type='Testing')
    return {'valid_loss': valid_loss.avg,
            'valid_acc': valid_acc.sum / count}


data = {
    'train': datasets.CIFAR10(
        root='./data', download=True,
        transform=transforms.Compose([
            # transforms.RandomCrop((32, 32), padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
    ),
    'valid': datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
    )
}

loaders = {
    'train': DataLoader(data['train'], batch_size=128, shuffle=True,
                        num_workers=10, pin_memory=True,
                        drop_last=True),
    'valid': DataLoader(data['valid'], batch_size=128,
                        num_workers=10, pin_memory=True,
                        drop_last=False)
}
parser = argparse.ArgumentParser(description='CIFAR10')
parser.add_argument("--model", type=str, default='ResNet18')
parser.add_argument('--drop_p', default=0.5, type=float)
parser.add_argument("--drop_num", default=0, type=int, help='number of multi sample dropout')
args = parser.parse_args()

model = MiniResNet(num_classes=10,dropout_num = args.drop_num,dropout_p=args.drop_p)
model.to(device)
arch = arch+f"_{args.drop_num}samples"
train_monitor = TrainingMonitor(file_dir='./', arch=arch)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(1, epochs + 1):
    train_log = train(loaders['train'])
    valid_log = test(loaders['valid'])
    logs = dict(train_log, **valid_log)
    show_info = f'\nEpoch: {epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
    print(show_info)
    if epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate * 0.1
    train_monitor.epoch_step(logs)
