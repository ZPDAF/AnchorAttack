import logging
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


def train_trigger(epoch, client1, server, client2, data_loader, optimizers, loss_function, device, num_class,
                  anchors, trigger):
    r = trigger
    train_loss = 0
    correct = 0
    iter_count = 0
    for _, (x, y) in enumerate(data_loader):
        for optimizer in optimizers:
            optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        embed_c2s = client1(x)
        embed_s2c_ = server(embed_c2s)
        embed_s2c_ = embed_s2c_.clone().detach()
        B, C, H, W = embed_c2s.size(0), embed_c2s.size(1), embed_c2s.size(2), embed_c2s.size(3)
        # 正常训练过程的损失函数
        y_pred = client2(server(embed_c2s))
        correct += y_pred.max(1)[1].eq(y).sum().item()
        loss = loss_function(y_pred, y)
        train_loss += loss
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        if epoch >= 0:
            # 生成r
            if r is None:
                r = torch.rand(1, C, H, W).to(device)

            # 随机挑选若干样本添加r
            l = list(range(B))
            list_slice = l
            embed_c2s_a = embed_c2s[list_slice[0]].clone().detach().unsqueeze(0)
            for i in range(1, len(list_slice)):
                embed_c2s_a = torch.cat((embed_c2s_a,
                                         embed_c2s[list_slice[i]].clone().detach().unsqueeze(0)))
            embed_c2s_a += r.clone().detach()
            embed_c2s_a.to(device)
            embed_c2s_a.requires_grad_()

            # 生成锚点
            if anchors is None:
                with torch.no_grad():
                    embed_sample = server(torch.cat(
                        (embed_c2s[0].clone().detach().unsqueeze(0), embed_c2s[0].clone().detach().unsqueeze(0))))
                C2, H2, W2 = embed_sample.size(1), embed_sample.size(2), embed_sample.size(3)
                anchors = torch.rand(num_class, C2, H2, W2)
                # 二值化
                # 生成一个 包含1到C2*H2*W2数值但顺序随机的序列，每次截取一段将一个全一[1,C2,H2,W2]tensor中的相应部分转为-1，然后作为anchor
                a = torch.ones_like(anchors)
                b = -torch.ones_like(anchors)
                z = torch.full_like(anchors, fill_value=0.5)
                anchors = torch.where(torch.ge(anchors, z), a, b)
                anchors.to(device)
                del embed_sample, a, b, z

            y_anchor = anchors[0].clone().detach().unsqueeze(0)
            for i in range(1, len(list_slice)):
                y_anchor = torch.cat((y_anchor, anchors[0].clone().detach().unsqueeze(0)))
            y_anchor = y_anchor.to(device)
            y_anchor.requires_grad_()

            for _ in range(2):
                embed_s2c_a = server(embed_c2s_a)
                loss_l2 = nn.MSELoss()
                # loss2 = 0.05 * loss_l2(embed_s2c_a, y_anchor)
                loss2 = 2 * loss_l2(embed_s2c_a, y_anchor)
                embed_c2s_t = embed_c2s.clone().detach()
                embed_s2c_t = server(embed_c2s_t)
                loss3 = loss_l2(embed_s2c_t, embed_s2c_.clone().detach())
                loss_total = loss2 + loss3
                loss_total.backward()
                optimizers[1].zero_grad()
                optimizers[1].step()

    num_data = len(data_loader.dataset)
    acc = correct / num_data
    train_loss = train_loss / num_data
    print(f'epoch：{epoch} 训练准确率：{acc:.4f} 训练损失：{train_loss:.6f}')
    logging.info("epoch：%d 训练准确率：%.4f 训练损失：%.6f", epoch, acc, train_loss)
    print(f'epoch：{epoch} 攻击迭代次数：{iter_count}')
    logging.info(f'epoch：{epoch} 攻击迭代次数：{iter_count}')
    return r, anchors


def train_surrogate(epoch, net, data_loader, optimizer, loss_f, device):
    total_loss = 0
    i = 0

    for _, (x, y) in enumerate(data_loader):
        i += 1
        x, y = x.to(device), y.to(device)

        lam = np.random.beta(0.2, 0.2)
        index = torch.randperm(x.size(0)).cuda()
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_pred = net(mixed_x)
        loss = lam * loss_f(y_pred, y) + (1 - lam) * loss_f(y_pred, y[index])
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'{total_loss * 1.0 / i:.4f}')
    logging.info(f'{total_loss * 1.0 / i:.4f}')


def train(epoch, client1, server, client2, data_loader, optimizers, loss_function, device):
    train_loss = 0
    correct = 0
    for _, (X, y) in enumerate(data_loader):
        for optimizer in optimizers:
            optimizer.zero_grad()

        y_true = y.to(device)
        y_pred = client2(server(client1(X.to(device))))

        loss = loss_function(y_pred, y_true)
        train_loss += loss
        loss.backward()

        correct += y_pred.max(1)[1].eq(y_true).sum().item()

        for optimizer in optimizers:
            optimizer.step()

    num_data = len(data_loader.dataset)
    acc = correct / num_data
    train_loss = train_loss / num_data
    print(f'epoch:{epoch} 训练准确率：{acc:.4f} 训练损失：{train_loss:.6f}')
    logging.info("epoch:%d 训练准确率：%.4f 训练损失：%.6f", epoch, acc, train_loss)
