import logging
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def test_attack(epoch, client1, server, client2, data_loader, loss_function, device, num_class, trigger_embed, anchors):
    with torch.no_grad():
        client1.eval()
        server.eval()
        client2.eval()

        test_loss = 0
        correct = 0
        attack_correct = 0
        num_except_attack = 1
        target_count = [0] * num_class
        true_count = [0] * num_class
        y_pred_t = client2(torch.cat((anchors[0].unsqueeze(0), anchors[0].unsqueeze(0))).to(device))
        y_pred_t = y_pred_t.max(1)[1]
        attack_class = y_pred_t[0]
        for _, (x, y) in enumerate(data_loader):
            y_true = y.to(device)
            y_pred = client1(x.to(device))

            if epoch >= 0:
                # 得出当前batch攻击成功数
                y_pred_r = y_pred
                y_pred_r += trigger_embed
                y_pred_r = client2(server(y_pred_r))
                y_pred_r = y_pred_r.max(1)[1]
                for j in range(y_pred_r.size(0)):
                    target_count[y_pred_r[j]] += 1
                    if y_pred_r[j] == y_true[j]:
                        true_count[y_pred_r[j]] += 1
                del y_pred_r

            # 得出当前batch模型预测成功数
            y_pred = client2(server(y_pred))
            loss = loss_function(y_pred, y_true)
            test_loss += loss
            correct += y_pred.max(1)[1].eq(y_true).sum().item()

        num_data = len(data_loader.dataset)
        if epoch >= 0:
            target_count = np.array(target_count)
            true_count = np.array(true_count)
            attack_count = target_count - true_count
            # attack_class = np.argmax(attack_count)
            num_except_attack = num_data - true_count[attack_class]
            attack_correct = attack_count[attack_class]
        acc = correct / num_data
        acc_attack = attack_correct / num_except_attack
        test_loss = test_loss / num_data
        print(f'epoch:{epoch} 攻击类别：{attack_class} 攻击准确率：{acc_attack:.4f}')
        print(f'epoch:{epoch} 测试准确率：{acc:.4f} 测试损失：{test_loss:.6f}')
        print("   ")
        logging.info(f'epoch:{epoch} 攻击类别：{attack_class} 攻击准确率：{acc_attack:.4f}')
        logging.info("epoch:%d 测试准确率：%.4f 测试损失：%.6f", epoch, acc, test_loss)
        logging.info(print("   "))


def test_surrogate(epoch, client1, server, net, data_loader, loss_function, device, num_class):
    with torch.no_grad():
        client1.eval()
        server.eval()
        net.eval()

        test_loss = 0
        correct = 0
        i = 0
        for _, (x, y) in enumerate(data_loader):
            i += 1
            y_true = y.to(device)
            y_pred = server(client1(x.to(device)))
            y_pred = net.net_surrogate(y_pred)
            y_pred = torch.squeeze(y_pred)

            loss = loss_function(y_pred, y_true)
            test_loss += loss

            correct += y_pred.max(1)[1].eq(y_true).sum().item()

        num_data = len(data_loader.dataset)
        acc = correct / num_data
        test_loss = test_loss / num_data
        print(f'epoch:{epoch} 测试准确率：{acc:.4f} 测试损失：{test_loss:.6f}')
        logging.info("epoch:%d 测试准确率：%.4f 测试损失：%.6f", epoch, acc, test_loss)


def mytest(epoch, client1, server, client2, data_loader, loss_function, device, num_class):
    def acc(y_true, y_pred):
        """
            Calculate clustering accuracy. Require scikit-learn installed
            # Arguments
                y: true labels, numpy.array with shape `(n_samples,)`
                y_pred: predicted labels, numpy.array with shape `(n_samples,)`
            # Return
                accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        ind = np.array(ind).T
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    with torch.no_grad():
        # Kmeans
        clt = KMeans(n_clusters=num_class)
        client1.eval()
        server.eval()
        client2.eval()

        test_loss = 0
        correct = 0

        test_correct_acc = 0
        i = 0
        for _, (X, y) in enumerate(data_loader):
            i += 1
            y_true = y.to(device)
            embed_c2s = client1(X.to(device))
            B = embed_c2s.size(0)
            embed_s2c = server(embed_c2s)
            y_pred_cluster = clt.fit_predict(embed_c2s.view(B, -1).cpu())
            # embed_s2c.isnull().sum().tolist()
            y_pred = client2(embed_s2c)

            loss = loss_function(y_pred, y_true)
            test_loss += loss

            correct_acc = acc(y_true.cpu().numpy(), y_pred_cluster)
            test_correct_acc += correct_acc

            correct += y_pred.max(1)[1].eq(y_true).sum().item()

        num_data = len(data_loader.dataset)
        test_correct_acc = test_correct_acc / i
        acc = correct / num_data
        test_loss = test_loss / num_data
        print(f'epoch:{epoch} 聚类准确率：{test_correct_acc:.4f} 测试准确率：{acc:.4f} 测试损失：{test_loss:.6f}')
        logging.info("epoch:%d 聚类准确率:%.4f 测试准确率：%.4f 测试损失：%.6f", epoch, test_correct_acc, acc, test_loss)
