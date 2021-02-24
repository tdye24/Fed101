import os
import copy
import torch
from torchvision import transforms
import numpy as np

from tensorboardX import SummaryWriter
from algorithm.FedSP.client import Client

from data.mnist.MNIST_DATASET import get_mnist_dataloaders
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders
from data.femnist.FEMNIST_DATASET import get_femnist_dataloaders


class Server:
    def __init__(self,
                 seed=123,
                 rounds=20,
                 epoch=1,
                 clients_per_round=1,
                 eval_interval=1,
                 dataset_name='femnist',
                 model_name='cnn',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 pretrain_model=None,
                 decay_step=200,
                 note=''):
        self.clients = []
        self.seed = seed  # randomly sampling

        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.batch_size = batch_size

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.global_params = None
        self.updates = []
        self.selected_clients = []
        self.have_seen_clients = set()  # user_id set, not client set, not numpy

        self.clients_per_round = clients_per_round
        self.epoch = epoch
        self.rounds = rounds

        self.pretrain_model = pretrain_model  # pre-trained model  directory
        self.eval_interval = eval_interval

        self.optim = {'round': 0,
                      'M1_acc': -1.0, 'M2_acc': -1.0,
                      'M1_params': None, 'M2_params': None,
                      'M1_loss': 10e8, 'M2_loss': 10e8}

        self.M1_train_writer = None
        self.M1_test_writer = None
        self.M2_train_writer = None
        self.M2_test_writer = None

        self.flag = None
        self.note = note

    def initiate(self):
        self.clients = self.setup_clients(self.dataset_name,
                                          model_name=self.model_name,
                                          lr=self.lr,
                                          batch_size=self.batch_size)
        assert self.clients_per_round <= len(self.clients)
        self.clients_per_round = min(self.clients_per_round, len(self.clients))
        if len(self.clients) > 0 and self.pretrain_model is None:
            self.global_params = copy.deepcopy(self.clients[0].get_global_params())
        else:
            print("Error：length of clients list is zero!")
            exit(0)
        batch_size = self.batch_size
        dataset_name = self.dataset_name
        model_name = self.model_name
        clients_per_round = self.clients_per_round
        epoch = self.epoch

        if batch_size >= self.clients[0].trainloader.sampler.num_samples:
            flag = "N"
        else:
            flag = batch_size
        self.flag = flag

        self.M1_train_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedsp/M1_{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{flag}_lr{self.lr}_train_{self.note}')
        self.M1_test_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedsp/M1_{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{flag}_lr{self.lr}_val_{self.note}')

        self.M2_train_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedsp/M2_{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{flag}_lr{self.lr}_train_{self.note}')
        self.M2_test_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedsp/M2_{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{flag}_lr{self.lr}_val_{self.note}')

    def setup_clients(self, dataset_name, model_name: str, batch_size: int, lr: float):
        users = []
        trainloaders, testloaders = [], []

        if self.dataset_name == 'cifar10':
            # data augmentation
            # train_transform = transforms.Compose([
            #     # transforms.RandomCrop(size=24, padding=8, fill=0, padding_mode='constant'),
            #     transforms.RandomHorizontalFlip(p=0.5),
            #     transforms.RandomApply([
            #         transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.8),
            #
            #     transforms.ToTensor(),
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                          std=[0.229, 0.224, 0.225])
            # ])
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            # TODO(specify the num of all clients: default 100 for cifar10 dataset)
            users, trainloaders, testloaders = get_cifar10_dataloaders(batch_size=self.batch_size,
                                                                       train_transform=train_transform,
                                                                       test_transform=test_transform)
        elif self.dataset_name == 'mnist':
            train_transform = None
            test_transform = None
            users, trainloaders, testloaders = get_mnist_dataloaders(batch_size=self.batch_size,
                                                                     train_transform=train_transform,
                                                                     test_transform=test_transform)
        elif self.dataset_name == 'femnist':
            users, trainloaders, testloaders = get_femnist_dataloaders(batch_size=self.batch_size)
        clients = [
            Client(user_id=user_id,
                   seed=self.seed,
                   trainloader=trainloaders[user_id],
                   testloader=testloaders[user_id],
                   model_name=model_name,
                   batch_size=batch_size,
                   lr=lr,
                   epoch=self.epoch,
                   lr_decay=self.lr_decay,
                   decay_step=self.decay_step) for user_id in users]
        return clients

    def select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False)
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
                # weight
                w = client_samples / total_weight
                if i == 0:
                    new_params[k] = client_params[k] * w
                else:
                    new_params[k] += client_params[k] * w
        # update global model params
        self.global_params = copy.deepcopy(new_params)

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
            self.select_clients(round_th=i)

            for c in self.selected_clients:
                c.set_global_params(self.global_params)
                num_train_samples, global_feature_update, loss = c.train(round_th=i)
                self.updates.append((num_train_samples, copy.deepcopy(global_feature_update)))

            # update have-seen client user_id set
            last_round_user_ids = [client.user_id for client in self.selected_clients]
            for user_id in last_round_user_ids:
                self.have_seen_clients.add(user_id)
            print(f"Have seen {len(self.have_seen_clients)}, {len(self.have_seen_clients) / len(self.clients)}")

            # average
            self.average()

            # clear
            self.updates = []
            self.selected_clients = []

            if i % self.eval_interval == 0:
                print("--------------------------\n")
                print("Round {}".format(i))
                # 2种测试方法
                # 1. 对于未见过的clients不测试
                # 2. 对于未见过的clients，用本地随机初始化的模型测试

                # TODO 方法1，对于未见过的clients不测试
                # test on training data
                for c in self.clients:
                    c.set_global_params(self.global_params)
                acc_over_all, loss_over_all = self.test(dataset='train', method='method1')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TRAIN# M1, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                self.M1_train_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.M1_train_writer.add_scalar('loss', avg_loss_all, global_step=i)
                # test on testing data
                acc_over_all, loss_over_all = self.test(dataset='test', method='method1')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TEST# M1, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                if avg_acc_all > self.optim['M1_acc']:
                    print("\033[1;31m" + "***M1***Best Model***SAVE***" + "\033[0m")
                    self.optim.update(
                        {'round': i, 'M1_acc': avg_acc_all, 'M1_params': self.global_params, 'M1_loss': avg_loss_all})
                    self.save_model(method='M1')

                self.M1_test_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.M1_test_writer.add_scalar('loss', avg_loss_all, global_step=i)

                # TODO 方法2，对于未见过的clients，用本地随机初始化的模型测试
                # test on training data
                # for c in self.clients:
                #     c.set_global_params(self)
                acc_over_all, loss_over_all = self.test(dataset='train', method='method2')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TRAIN# M2, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                self.M2_train_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.M2_train_writer.add_scalar('loss', avg_loss_all, global_step=i)
                # test on testing data
                acc_over_all, loss_over_all = self.test(dataset='test', method='method2')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TEST# M2, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                if avg_acc_all > self.optim['M2_acc']:
                    print("\033[1;31m" + "***M2***Best Model***SAVE***" + "\033[0m")
                    self.optim.update(
                        {'round': i, 'M2_acc': avg_acc_all, 'M2_params': self.global_params, 'M2_loss': avg_loss_all})
                    self.save_model(method='M2')

                self.M2_test_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.M2_test_writer.add_scalar('loss', avg_loss_all, global_step=i)

    def test(self, dataset='test', method='method1'):
        acc_list, loss_list = [], []

        # 方法1，对于未见过的clients不测试
        if method == 'method1':
            for c in self.clients:
                if c.user_id in self.have_seen_clients:
                    num_test_samples, acc, loss = c.test(dataset=dataset)
                    acc_list.append((num_test_samples, acc))
                    loss_list.append((num_test_samples, loss))
        # 方法2，对于未见过的clients，用本地随机初始化的模型测试
        elif method == 'method2':
            for c in self.clients:
                num_test_samples, acc, loss = c.test(dataset=dataset)
                acc_list.append((num_test_samples, acc))
                loss_list.append((num_test_samples, loss))
        return acc_list, loss_list

    def print_optim(self):
        print("Round {}".format(self.rounds), end=' ')
        # TODO 方法1，对于未见过的clients不测试
        # test on training data
        for c in self.clients:
            c.set_global_params(self)
        acc_over_all, loss_over_all = self.test(dataset='train', method='method1')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
        print("#TRAIN# M1, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        self.M1_train_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.M1_train_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)
        # test on testing data
        acc_over_all, loss_over_all = self.test(dataset='test', method='method1')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
        print("#TEST# M1, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        self.M1_test_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.M1_test_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)

        if avg_acc_all > self.optim['M1_acc']:
            print("\033[1;31m" + "***M1***Best Model***SAVE***" + "\033[0m")
            self.optim.update(
                {'round': self.rounds, 'M1_acc': avg_acc_all, 'M1_params': self.global_params, 'M1_loss': avg_loss_all})
            self.save_model(method='M1')

        # TODO 方法2，对于未见过的clients，用本地随机初始化的模型测试
        # test on training data
        # for c in self.clients:
        #     c.set_global_params(self)
        acc_over_all, loss_over_all = self.test(dataset='train', method='method2')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
        print("#TRAIN# M2, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        self.M2_train_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.M2_train_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)

        # test on testing data
        acc_over_all, loss_over_all = self.test(dataset='test', method='method2')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
        print("#TEST# M2, Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        if avg_acc_all > self.optim['M2_acc']:
            print("\033[1;31m" + "***M2***Best Model***SAVE***" + "\033[0m")
            self.optim.update(
                {'round': self.rounds, 'M2_acc': avg_acc_all, 'M2_params': self.global_params, 'M2_loss': avg_loss_all})
            self.save_model(method='M2')

        self.M2_test_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.M2_test_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)

        print("\n")
        print(
            f"###### M1-Round: {self.optim['round']}, Optimal Federated Model, Average Accuracy Over All Clients(AAAC): "
            f"\033[1;32m{self.optim['M1_acc']}\033[0m######")
        print(
            f"###### M2-Round: {self.optim['round']}, Optimal Federated Model, Average Accuracy Over All Clients(AAAC): "
            f"\033[1;32m{self.optim['M2_acc']}\033[0m######")

    def save_model(self, method='M1'):
        path = f'/home/tdye/Fed101/result/fedsp/{self.dataset_name}_{self.model_name}_C{self.clients_per_round}_E{self.epoch}_B{self.flag}_lr{self.lr}_{self.note}'
        if not os.path.exists(path):
            os.makedirs(path)
        path = f'{path}/model-{method}.pkl'
        print(f"model saved to：{path}")
        torch.save(self.global_params, path)
