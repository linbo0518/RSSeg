import os
from datetime import datetime

import torch
from torch import optim
from torch.utils import data

from dataset import RSClsDataset
from model import ResNet34
from loss import SoftLabelCrossEntropyLoss
from cls_utils import train, valid, save_model

root_dir = '/path/to/root/'
train_csv = os.path.join(root_dir, 'data/train/desc.csv')
valid_csv = os.path.join(root_dir, 'data/valid/desc.csv')
train_image_dir = os.path.join(root_dir, 'data/train/image/')
valid_image_dir = os.path.join(root_dir, 'data/valid/image/')
device = 1
lr = 1e-2
batch_size = 32
epochs = 60
print_interval = 20
save_model_dir = os.path.join(root_dir, 'model/')
pretrained_params = 'resnet34_params.pt'

print('%s Start Preprocessing' % (datetime.now().ctime()))
train_dataset = RSClsDataset(train_csv, train_image_dir, 'train')
train_dataloader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
valid_dataset = RSClsDataset(valid_csv, valid_image_dir, 'valid')
valid_dataloader = data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

net = ResNet34().cuda(device)
net.load_state_dict(
    torch.load(os.path.join(save_model_dir, pretrained_params)), strict=False)

criterion = SoftLabelCrossEntropyLoss().cuda(device)

optimizer = optim.SGD(
    net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[10, 30, 50], gamma=0.1)
print('%s Finish Preprocessing' % (datetime.now().ctime()))

print('%s Start Training' % (datetime.now().ctime()))
print(
    'Dtype[%s] Base lr[%f] Epochs[%d] Dataset: train[%d] valid[%d] Iter: train[%d] valid[%d] Batch Size[%d]'
    % (str(torch.get_default_dtype()), optimizer.param_groups[0]['lr'], epochs,
       len(train_dataset), len(valid_dataset), len(train_dataloader),
       len(valid_dataloader), batch_size))

valid(valid_dataloader, net, criterion, device)

for epoch in range(epochs):
    scheduler.step()
    train(train_dataloader, net, criterion, device, epoch, optimizer,
          batch_size, print_interval)
    valid_detail = valid(valid_dataloader, net, criterion, device)
    save_model(net, save_model_dir, 'cls', epoch, valid_detail)

print('%s Finish Training' % (datetime.now().ctime()))