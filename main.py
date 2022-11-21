import argparse
import os
import numpy as np
import torchvision
from sklearn import metrics
from sklearn.cluster import KMeans
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image
from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, MNIST
import tqdm
from tqdm import trange
import logging
from src import pwd
from src.model import NetClient1, NetServer, NetClient2, NetSurrogate
from src.train import train_surrogate, train, train_trigger
from src.test import test_surrogate, mytest, test_attack

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

datasets_dict = {'mnist': torchvision.datasets.MNIST,
                 'cifar10': torchvision.datasets.CIFAR10}
datasets_num_dict = {'mnist': 10,
                     'cifar10': 10}
datasets_in_channels = {'mnist': 1,
                        'cifar10': 3}


class MyDataSet(Dataset):
    def __init__(self, imgs, labels):
        self._data = imgs
        self._label = labels

    def __getitem__(self, idx):
        img = self._data[idx]
        label = self._label[idx]
        return img, label

    def __len__(self):
        return len(self._data)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    num_class_ = datasets_num_dict[args.dataset]
    task_path = os.path.join(pwd, "log", args.dataset)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
    logging.basicConfig(filename=os.path.join(task_path, "run.log"),
                        format='[%(asctime)s %(filename)s] %(levelname)s:%(message)s', level=logging.DEBUG,
                        filemode='a', datefmt='%Y-%m-%d  %I:%M:%S')
    # Net
    net1 = NetClient1(datasets_in_channels[args.dataset], num_class_).cuda()
    net2 = NetServer(num_class_).cuda()
    net3 = NetClient2(num_class_).cuda()
    net_s = NetSurrogate(datasets_in_channels[args.dataset], num_class_).cuda()
    # loss
    loss_ = nn.CrossEntropyLoss().cuda()
    # Optimizer itertools.chain(?)
    optimizers_ = [torch.optim.Adam(net1.parameters(), lr=args.lr, weight_decay=args.wd),
                   torch.optim.Adam(net2.parameters(), lr=args.lr, weight_decay=args.wd),
                   torch.optim.Adam(net3.parameters(), lr=args.lr, weight_decay=args.wd)]
    optimizer_s = torch.optim.Adam(net_s.parameters(), lr=args.lr, weight_decay=args.wd)
    # transformer
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) if (
            args.dataset == 'mnist') \
        else transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # dataset
    train_data = datasets_dict[args.dataset](root='./data', train=True, download=True, transform=transformer)
    test_data = datasets_dict[args.dataset](root='./data', train=False, download=True, transform=transformer)

    data_sub = []
    label_sub = []
    count_class = [0] * 10
    num_per_class = 10
    for x, y in train_data:
        if count_class[y] < num_per_class:
            data_sub.append(x)
            label_sub.append(y)
            count_class[y] += 1
        if count_class == [num_per_class] * 10:
            break
    train_data_sub = MyDataSet(data_sub, label_sub)

    # DataLoader
    train_dataloader = DataLoader(train_data, batch_size=32)
    test_dataloader = DataLoader(test_data, batch_size=32)
    # train_dataloader_sub = DataLoader(train_data_sub, batch_size=3, shuffle=True, drop_last=True)
    device_ = args.device

    # if args.surrogate:
    #     for epoch_ in range(100):
    #         print(f'替代模型第{epoch_+1}轮训练')
    #         logging.info(f'替代模型第{epoch_+1}轮训练')
    #         train_surrogate(epoch_, net_s, train_dataloader_sub, optimizer_s, loss_, device_)

    anchors_, trigger_ = None, None
    for epoch_ in range(args.epochs):
        # train(epoch_, net1, net2, net3, train_dataloader, optimizers_, loss_, device_)
        trigger_, anchors_ = train_trigger(epoch_, net1, net2, net3, train_dataloader, optimizers_, loss_, device_,
                                           num_class_, anchors_, trigger_)
        if not args.surrogate:
            # mytest(epoch_, net1, net2, net3, test_dataloader, loss_, device_, num_class_)
            test_attack(epoch_, net1, net2, net3, test_dataloader, loss_, device_, num_class_, trigger_, anchors_)
        else:
            test_surrogate(epoch_, net1, net2, net_s, test_dataloader, loss_, device_, num_class_)

    # net_s.state_dict()["resnet.weight"].copy_(net2.state_dict()["resnet_medium.weight"].clone().detach())
    # net_s.resnet.load_state_dict(net2.resnet_medium.state_dict())
    # for epoch_ in range(100):
    #     if args.surrogate:
    #         print(f'替代模型第{epoch_ + 1}轮训练')
    #         logging.info(f'替代模型第{epoch_ + 1}轮训练')
    #         train_surrogate(epoch_, net_s, train_dataloader_sub, optimizer_s, loss_, device_)
    # test_surrogate(epoch_, net1, net2, net_s, test_dataloader, loss_, device_, num_class_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--device', type=int, default=0, help='using device')
    parser.add_argument('--dataset', type=str, default='mnist', help='using dataset')
    parser.add_argument('--surrogate', type=bool, default=False, help='using surrogate model or using clustering')
    args_ = parser.parse_args()
    main(args_)
