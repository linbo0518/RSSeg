import os
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F


class MAE:

    def __init__(self):
        self._some_metric = 0
        self._num_sample = 0

    def reset(self):
        self._some_metric = 0
        self._num_sample = 0

    def update(self, pred, label):
        assert pred.shape == label.shape
        pred = F.softmax(pred, dim=-1)
        self._some_metric += (pred - label).abs().sum().item()
        self._num_sample += pred.size(0)

    def get(self):
        return self._some_metric / self._num_sample


def train(dataloader, net, criterion, device, epoch, optimizer, batch_size,
          print_interval):
    net.train()
    cls_loss = []
    cls_mae = MAE()
    start_time = datetime.now().timestamp()

    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        image = batch_data[0].cuda(device)
        label = batch_data[1].cuda(device)

        pred = net(image)

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cls_loss.append(loss.item())
            cls_mae.update(pred, label)

        if batch_idx % print_interval == print_interval - 1:
            speed = batch_size * print_interval / (
                datetime.now().timestamp() - start_time)
            train_detail = {
                'cls_loss': np.mean(cls_loss),
                'cls_mae': cls_mae.get(),
            }

            print(
                '\nepoch[%02d] batch[%04d] lr[%f] speed[%f(sample/sec)] time[%s]'
                % (epoch + 1, batch_idx + 1, optimizer.param_groups[0]['lr'],
                   speed, datetime.now().ctime()))
            print_detail('train', train_detail)

            cls_loss = []
            cls_mae.reset()
            start_time = datetime.now().timestamp()


def valid(dataloader, net, criterion, device):
    net.eval()
    cls_loss = []
    cls_mae = MAE()

    with torch.no_grad():
        for batch_data in dataloader:
            image = batch_data[0].cuda(device)
            label = batch_data[1].cuda(device)

            pred = net(image)

            loss = criterion(pred, label)

            cls_loss.append(loss.item())
            cls_mae.update(pred, label)

    valid_detail = {
        'cls_loss': np.mean(cls_loss),
        'cls_mae': cls_mae.get(),
    }
    print_detail('valid', valid_detail)
    return valid_detail


def print_detail(mode, detail):
    assert mode == 'train' or mode == 'valid'
    print('%s loss:' % (mode))
    print('  cls_loss: %5f' % (detail['cls_loss']))
    print('%s metric:' % (mode))
    print('  cls_mae: %5f' % (detail['cls_mae']))


def save_model(net, save_model_dir, prefix, epoch, valid_detail):
    current_datetime = datetime.now()
    datetime_prefix = '{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}'.format(
        current_datetime.year, current_datetime.month, current_datetime.day,
        current_datetime.hour, current_datetime.minute, current_datetime.second)
    model_filename = '{}_{}_epoch[{:0>2d}]_mae[{:.5f}].pth'.format(
        datetime_prefix, prefix, epoch + 1, valid_detail['cls_mae'])
    torch.save(net.state_dict(), os.path.join(save_model_dir, model_filename))
    print('model params saved to %s' % (os.path.join(save_model_dir,
                                                     model_filename)))
