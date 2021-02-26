import os
import copy
import torch
from torchvision import transforms
import numpy as np

from algorithm.BASE import BASE
from tensorboardX import SummaryWriter
from algorithm.FedPROX.client import Client

# dataloaders
from data.mnist.MNIST_DATASET import get_mnist_dataloaders
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders
from data.femnist.FEMNIST_DATASET import get_femnist_dataloaders

from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


class Server(BASE):
    def __init__(self,
                 seed=123,
                 rounds=20,
                 epoch=1,
                 clients_per_round=1,
                 eval_interval=1,
                 dataset_name='femnist',
                 model_name='femnist',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 decay_step=200,
                 note=''):
        BASE.__init__(self, algorithm='fedprox', seed=seed, epoch=epoch, model_name=model_name,
                      lr=lr, batch_size=batch_size, lr_decay=lr_decay, decay_step=decay_step)

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.params = self.model.state_dict()
        self.updates = []
        self.selected_clients = []
        self.clients_per_round = clients_per_round
        self.rounds = rounds
        self.eval_interval = eval_interval
        self.note = note

        self.optim = {'round': 0, 'acc': -1.0, 'params': None, 'loss': 10e8}

        self.train_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedprox/{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{batch_size}_lr{lr}_train_{note}')
        self.test_writer = SummaryWriter(
            f'/home/tdye/Fed101/visualization/fedprox/{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{batch_size}_lr{lr}_val_{note}')

        self.clients = self.setup_clients()
        assert self.clients_per_round <= len(self.clients)

        self.surrogates = self.setup_surrogates()
        assert len(self.surrogates) == clients_per_round

    def setup_clients(self):
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
                   model_name=self.model_name,
                   lr=self.lr,
                   epoch=self.epoch,
                   lr_decay=self.lr_decay,
                   decay_step=self.decay_step)
            for user_id in users]
        return clients

    def setup_surrogates(self):
        surrogates = [
            Client(user_id=i,
                   seed=self.seed,
                   trainloader=None,
                   testloader=None,
                   model_name=self.model_name,
                   lr=self.lr,
                   epoch=self.epoch,
                   lr_decay=self.lr_decay,
                   decay_step=self.decay_step)
            for i in range(self.clients_per_round)]
        return surrogates

    def select_clients(self, round_th):
        np.random.seed(seed=self.seed + round_th)
        selected_clients = np.random.choice(self.clients, self.clients_per_round, replace=False)
        self.selected_clients = selected_clients

    def average(self):
        updates = self.updates
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
        self.params = new_params

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

            for k in range(len(self.selected_clients)):
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_params(self.params)
                num_train_samples, update, loss = surrogate.train(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((num_train_samples, copy.deepcopy(update)))

            # average
            self.average()

            # clear
            self.updates = []
            self.selected_clients = []

            if i % self.eval_interval == 0:
                print("--------------------------\n")
                print("Round {}".format(i))
                # test on training data
                acc_over_all, loss_over_all = self.test(dataset='train')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TRAIN# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                self.train_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.train_writer.add_scalar('loss', avg_loss_all, global_step=i)
                # test on testing data
                acc_over_all, loss_over_all = self.test(dataset='test')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TEST# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                if avg_acc_all > self.optim['acc']:
                    print("\033[1;31m" + "***Best Model***SAVE***" + "\033[0m")
                    self.optim.update({'round': i, 'acc': avg_acc_all, 'params': self.params, 'loss': avg_loss_all})
                    # self.save_model()

                self.test_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.test_writer.add_scalar('loss', avg_loss_all, global_step=i)

    def test(self, dataset='test'):
        acc_list, loss_list = [], []
        surrogate = self.surrogates[0]
        for c in self.clients:
            surrogate.update(c)
            surrogate.set_params(self.params)
            num_test_samples, acc, loss = surrogate.test(dataset=dataset)
            acc_list.append((num_test_samples, acc))
            loss_list.append((num_test_samples, loss))
        return acc_list, loss_list

    def print_optim(self):
        for c in self.clients:
            c.set_params(self.params)
        print("Round {}".format(self.rounds), end=' ')

        acc_all, loss_all = self.test(dataset='train')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_all), self.avg_metric(loss_all)

        print("#TRAIN# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        self.train_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.train_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)

        acc_all, loss_all = self.test(dataset='test')
        avg_acc_all, avg_loss_all = self.avg_metric(acc_all), self.avg_metric(loss_all)

        print("#TEST# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

        self.test_writer.add_scalar('acc', avg_acc_all, global_step=self.rounds)
        self.test_writer.add_scalar('loss', avg_loss_all, global_step=self.rounds)

        if avg_acc_all > self.optim['acc']:
            print("\033[1;31m" + "***Best Model***SAVE***" + "\033[0m")
            self.optim.update({'round': self.rounds, 'acc': avg_acc_all, 'params': self.params, 'loss': avg_loss_all})
            # self.save_model()
        print("\n")
        print(
            f"######Round: {self.optim['round']}, Optimal Federated Model, Average Accuracy Over All Clients(AAAC): "
            f"\033[1;32m{self.optim['acc']}\033[0m######")

    def save_model(self):
        path = f'/home/tdye/Fed101/result/fedprox/{self.dataset_name}_{self.model_name}_C{self.clients_per_round}_E{self.epoch}_B{self.batch_size}_lr{self.lr}_{self.note}'
        if not os.path.exists(path):
            os.makedirs(path)
        path = f'{path}/model.pkl'
        print(f"model saved to：{path}")
        torch.save(self.params, path)
