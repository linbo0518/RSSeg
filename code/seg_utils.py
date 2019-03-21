import os
from datetime import datetime
import numpy as np
import torch


def thresholding(pred, threshold, value=1):
    pred[pred > threshold] = value
    pred[pred <= threshold] = 0
    return pred


def evaluate(pred, label, threshold=0.5):
    eps = 1e-12
    pred = torch.sigmoid(pred)
    pred = thresholding(pred, threshold)

    pred = pred.byte()
    label = label.byte()
    intersection = (pred & label).float().sum((1, 2, 3))
    union = (pred | label).float().sum((1, 2, 3))
    iou = intersection / (union + eps)

    correct = (pred == label).float().sum()
    acc = correct / label.nelement()

    return iou.mean().item(), acc.item()


def train(dataloader, net, criterion, device, epoch, optimizer, batch_size,
          print_interval):
    net.train()
    seg_loss = []
    seg_iou = []
    seg_acc = []
    start_time = datetime.now().timestamp()

    for batch_idx, batch_data in enumerate(dataloader):
        optimizer.zero_grad()

        image = batch_data[0].cuda(device)
        mask = batch_data[1].cuda(device)

        pred = net(image)

        loss = criterion(pred, mask)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            iou, acc = evaluate(pred, mask)
            seg_loss.append(loss.item())
            seg_iou.append(iou)
            seg_acc.append(acc)

        if batch_idx % print_interval == print_interval - 1:
            speed = batch_size * print_interval / (
                datetime.now().timestamp() - start_time)
            train_detail = {
                'seg_loss': np.mean(seg_loss),
                'seg_iou': np.mean(seg_iou),
                'seg_acc': np.mean(seg_acc),
            }

            print(
                '\nepoch[%02d] batch[%04d] lr[%f] speed[%f(sample/sec)] time[%s]'
                % (epoch + 1, batch_idx + 1, optimizer.param_groups[0]['lr'],
                   speed, datetime.now().ctime()))
            print_detail('train', train_detail)

            seg_loss = []
            seg_iou = []
            seg_acc = []
            start_time = datetime.now().timestamp()


def valid(dataloader, net, criterion, device):
    net.eval()
    seg_loss = []
    seg_iou = []
    seg_acc = []

    with torch.no_grad():
        for batch_data in dataloader:
            image = batch_data[0].cuda(device)
            mask = batch_data[1].cuda(device)

            pred = net(image)

            loss = criterion(pred, mask)

            iou, acc = evaluate(pred, mask)
            seg_loss.append(loss.item())
            seg_iou.append(iou)
            seg_acc.append(acc)

    valid_detail = {
        'seg_loss': np.mean(seg_loss),
        'seg_iou': np.mean(seg_iou),
        'seg_acc': np.mean(seg_acc),
    }
    print_detail('valid', valid_detail)
    return valid_detail


def print_detail(mode, detail):
    assert mode == 'train' or mode == 'valid'
    print('%s loss:' % (mode))
    print('  seg_loss: %5f' % (detail['seg_loss']))
    print('%s metric:' % (mode))
    print('  seg_iou: %5f' % (detail['seg_iou']))
    print('  seg_acc: %5f' % (detail['seg_acc']))


def save_model(net, save_model_dir, prefix, epoch, valid_detail):
    current_datetime = datetime.now()
    datetime_prefix = '{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}'.format(
        current_datetime.year, current_datetime.month, current_datetime.day,
        current_datetime.hour, current_datetime.minute, current_datetime.second)
    model_filename = '{}_{}_epoch[{:0>2d}]_iou[{:.5f}].pth'.format(
        datetime_prefix, prefix, epoch + 1, valid_detail['seg_iou'])
    torch.save(net.state_dict(), os.path.join(save_model_dir, model_filename))
    print('model params saved to %s' % (os.path.join(save_model_dir,
                                                     model_filename)))