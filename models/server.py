import gc
import os
import copy
import sys
import torch.multiprocessing as mp
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from models.utils.model_utils import read_data, compare_models, Logger
# noinspection PyUnresolvedReferences
from tensorboardX import SummaryWriter
from models.client import Client
from data.CONSTANTS import *
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders
from data.mnist.MNIST_DATASET import get_mnist_dataloaders, get_mnist_dataloaders_niid

plt.switch_backend('agg')


class Server:
    def __init__(self, algorithm='fedavg', partitioning='iid', seed=123, rounds=20, epoch=1, all_clients_num=100,
                 clients_per_round=1,
                 eval_interval=1,
                 dataset_name='femnist', model_name='cnn', lr=3e-4, batch_size=1, mini_batch=0.1, lr_decay=0.99,
                 pretrain_model=None, decay_step=200):
        self.algorithm = algorithm
        self.clients = []
        self.seed = seed  # randomly sampling
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.mini_batch = mini_batch
        self.epoch = epoch
        self.partitioning = partitioning
        self.model_name = model_name
        self.rounds = rounds
        self.batch_size = batch_size
        self.params = None
        self.all_clients_num = all_clients_num
        self.clients_per_round = clients_per_round
        self.pretrain_model = pretrain_model  # 预训练模型地址
        self.eval_interval = eval_interval
        self.dataset_name = dataset_name
        self.updates = []
        self.selected_clients = []
        self.optim = {'round': 0, 'acc': 0.0, 'params': None, 'loss': 10e8}  # 第几轮，准确率，最高准确率对应的参数
        self.train_writer = None
        self.test_writer = None

    def initiate(self):
        self.clients = self.setup_clients(self.dataset_name, model_name=self.model_name, lr=self.lr,
                                          batch_size=self.batch_size,
                                          mini_batch=self.mini_batch)
        assert self.clients_per_round <= len(self.clients)
        self.clients_per_round = min(self.clients_per_round, len(self.clients))
        if len(self.clients) > 0 and self.pretrain_model is None:
            self.params = copy.deepcopy(self.clients[0].model.state_dict())
        elif self.pretrain_model is not None:
            self.load_pretrain()
        else:
            print("Error：clients长度为0")
            exit(0)
        batch_size = self.batch_size
        mini_batch = self.mini_batch
        dataset_name = self.dataset_name
        partitioning = self.partitioning
        model_name = self.model_name
        clients_per_round = self.clients_per_round
        epoch = self.epoch

        # 绘图初始化
        if batch_size >= len(self.clients[0].trainloader.dataset._ids):  # 全部数据作为一个batch
            flag = "NAN"
        else:
            flag = batch_size

        # 画训练和测试时的acc和loss，设置两个writer，主要是想让两个曲线（训练和验证）放在一起
        if mini_batch == -1:
            self.train_writer = SummaryWriter(
                f'../visualization/{self.algorithm}-{dataset_name}-{partitioning}-{model_name}-C{clients_per_round}-E{epoch}-B{flag}-train')
            self.test_writer = SummaryWriter(
                f'../visualization/{self.algorithm}-{dataset_name}-{partitioning}-{model_name}-C{clients_per_round}-E{epoch}-B{flag}-val')
        else:
            self.train_writer = SummaryWriter(
                f'../visualization/{self.algorithm}-{dataset_name}-{partitioning}-{model_name}-C{clients_per_round}-E{epoch}-M{mini_batch}-train')
            self.test_writer = SummaryWriter(
                f'../visualization/{self.algorithm}-{dataset_name}-{partitioning}-{model_name}-C{clients_per_round}-E{epoch}-M{mini_batch}-val')

    def load_pretrain(self):
        print(f"loading pre-trained model from {self.pretrain_model}")
        self.params = torch.load(self.pretrain_model)

    def setup_clients(self, dataset_name, model_name: str, batch_size: int, mini_batch: float, lr: float):
        # 改成dataloader形式
        # train_data_dir = os.path.join('..', 'data', dataset_name, 'data', self.partitioning, 'train')
        # test_data_dir = os.path.join('..', 'data', dataset_name, 'data', self.partitioning, 'test')
        #
        # users, train_data, test_data = read_data(train_data_dir, test_data_dir)
        users = []
        trainloaders, testloaders = [], []

        if self.dataset_name == 'cifar10':
            users = [i for i in range(100)]
            train_transform = transforms.Compose([
                # transforms.RandomCrop(size=24, padding=8, fill=0, padding_mode='constant'),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),

                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # train_transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225])
            # ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            trainloaders, testloaders = get_cifar10_dataloaders(batch_size=self.batch_size,
                                                                num_clients=self.all_clients_num,
                                                                train_transform=train_transform,
                                                                test_transform=test_transform)
        elif self.dataset_name == 'mnist':
            if self.partitioning == 'iid':
                users = [i for i in range(100)]
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])

                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])
                trainloaders, testloaders = get_mnist_dataloaders(batch_size=self.batch_size,
                                                                  num_clients=self.all_clients_num,
                                                                  train_transform=train_transform,
                                                                  test_transform=test_transform)
            else:
                users = [i for i in range(100)]
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])

                test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],
                                         std=[0.229])
                ])
                trainloaders, testloaders = get_mnist_dataloaders_niid(batch_size=self.batch_size,
                                                                       num_clients=self.all_clients_num,
                                                                       train_transform=train_transform,
                                                                       test_transform=test_transform)

        clients = [
            Client(user_id, trainloader=trainloaders[user_id], testloader=testloaders[user_id], model_name=model_name,
                   batch_size=batch_size,
                   mini_batch=mini_batch, lr=lr, epoch=self.epoch, lr_decay=self.lr_decay, decay_step=self.decay_step,
                   algorithm=self.algorithm) for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=round_th)
        selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False)
        #   设置client的随机种子，用于shuffle data(单独给每个client设置一个seed会导致每次shuffle的结果不变)
        for client in selected_clients:
            client.seed = round_th

        self.selected_clients = selected_clients

    def average(self):
        updates = copy.deepcopy(self.updates)
        total_weight = 0
        (client_samples, new_params) = updates[0]

        for (client_samples, client_params) in updates:
            total_weight += client_samples

        for k in new_params.keys():
            for i in range(0, len(updates)):
                client_samples, client_params = updates[i]
                # 权重
                w = client_samples / total_weight
                if i == 0:
                    new_params[k] = client_params[k] * w
                else:
                    new_params[k] += client_params[k] * w
        # 更新global模型参数
        self.params = copy.deepcopy(new_params)

    @staticmethod
    def avg_metric(metric_list):
        total_weight = 0
        total_metric = 0
        for (num_samples, metric) in metric_list:
            total_weight += num_samples
            total_metric += num_samples * metric
        avg_metric = total_metric / total_weight

        return avg_metric

    def federate(self):
        print("Begin Federating!")
        print(f"Training {len(self.clients)} clients!")
        for i in range(self.rounds):
            # 该条语句不能往后放，因为接下来要测试模型在选中的clients上的性能
            self.select_clients(round_th=i)
            if i % self.eval_interval == 0:
                print("--------------------------\n")
                print("Round {}".format(i))
                for c in self.clients:
                    c.set_params(self.params)
                acc_over_all, loss_over_all = self.test(dataset='test')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)

                if avg_acc_all > self.optim['acc']:
                    print("\033[1;31m" + "***当前最优模型***SAVE***" + "\033[0m")
                    self.optim.update({'round': i, 'acc': avg_acc_all, 'params': self.params, 'loss': avg_loss_all})
                    self.save_model()

                self.test_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.test_writer.add_scalar('loss', avg_loss_all, global_step=i)

                # 算一下在训练集上的表现
                for c in self.clients:
                    c.set_params(self.params)
                acc_over_all, loss_over_all = self.test(dataset='train')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)

                self.train_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.train_writer.add_scalar('loss', avg_loss_all, global_step=i)

            for c in self.selected_clients:
                c.set_params(self.params)
                num_train_samples, update, loss = c.train(round_th=i)
                self.updates.append((num_train_samples, copy.deepcopy(update)))

            # 联邦平均
            self.average()
            # 这一步本不需要再前向传播计算一遍loss，为了代码上的统一（和test），就再计算一遍吧

            # 清空本轮状态
            self.updates.clear()
            self.selected_clients = []

    def test(self, dataset='test'):
        """
        返回所有clients的平均准确率
        :param dataset: 在训练集上计算loss和acc还是在测试集上计算loss和acc
        :param domain: 参数是all,平均self.clients,参数是sub,平均self.selected_clients
        :return: 准确率列表，格式[（num_samples, acc）,...]
        """
        acc_list, loss_list = [], []

        for c in self.clients:
            num_test_samples, acc, loss = c.test(dataset=dataset)
            acc_list.append((num_test_samples, acc))
            loss_list.append((num_test_samples, loss))
            torch.cuda.empty_cache()
        if dataset == 'test':
            print(
                f"#TEST# All clients, Avg acc: {self.avg_metric(acc_list)}, Avg loss: {self.avg_metric(loss_list)}")
        else:
            print(
                f"#TRAIN# All clients, Avg acc: {self.avg_metric(acc_list)}, Avg loss: {self.avg_metric(loss_list)}")
        return acc_list, loss_list

    def print_optim(self):
        for c in self.clients:
            c.set_params(self.params)  # 下发最终的模型
        print("Round {}".format(self.rounds), end=' ')
        acc_all, loss_all = self.test(dataset='test')
        # acc_sub = self.test(domain="sub")
        avg_acc_all, avg_loss_all = self.avg_metric(acc_all), self.avg_metric(loss_all)
        # avg_acc_sub = self.avg_metric(acc_sub)

        if avg_acc_all > self.optim['acc']:
            print("\033[1;31m" + "***当前最优模型***SAVE***" + "\033[0m")
            self.optim.update({'round': self.rounds, 'acc': avg_acc_all, 'params': self.params, 'loss': avg_loss_all})
            self.save_model()
        print(
            f"######Round: {self.optim['round']}, Optimal Federated Model, Average Accuracy Over All Clients(AAAC): "
            f"\033[1;32m{self.optim['acc']}\033[0m######")

    def save_model(self):
        path = f'../result/{self.dataset_name}/{self.partitioning}/{self.model_name}/C{self.clients_per_round}-E{self.epoch}-B{self.batch_size}'
        if not os.path.exists(path):
            os.makedirs(path)
        path = f'{path}/model.pkl'
        print(f"模型已保存至：{path}")
        torch.save(self.params, path)
