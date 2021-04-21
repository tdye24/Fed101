import torch
import copy
import numpy as np
from algorithm.BASE import BASE
from tensorboardX import SummaryWriter

from algorithm.FedAVG.client import Client as FedAVG_Client
from algorithm.FedPROX.client import Client as FedPROX_Client
from algorithm.FedPER.client import Client as FedPER_Client
from algorithm.FedSP.client import Client as FedSP_Client
from algorithm.FedMoE.client import Client as FedMoE_Client
from algorithm.FedMC.client import Client as FedMC_Client
from algorithm.FedMC_WO.client import Client as FedMC_WO_Client
from algorithm.FedLG.client import Client as FedLG_Client

from torchvision.transforms import transforms

# data
# femnist
from data.femnist.FEMNIST_DATASET import get_femnist_dataloaders
# cifar10
from data.cifar10.CIFAR10_DATASET import get_cifar10_dataloaders
# cifar10_dirichlet
from data.cifar10.CIFAR10_DIRI import get_cifar10_dirichlet_dataloaders
# cifar10_latent_distribution
from data.cifar10.CIFAR10_LD import get_cifar10_dataloaders as get_cifar10_ld_dataloaders

# cifar100
from data.cifar100.CIFAR100_DATASET import get_cifar100_dataloaders
# cifar100 superclass
from data.cifar100.CIFAR100_SUPERCLASS import get_cifar100_dataloaders as get_cifar100_superclass_dataloaders

# mnist_malposition
from data.mnist.MNIST_DATASET_malposition import get_mnist_dataloaders as get_mnist_malposition_dataloaders
# mnist_wo_malposition
from data.mnist.MNIST_DATASET_wo_malposition import get_mnist_dataloaders as get_mnist_wo_malposition_dataloaders
# mnist_fedml
from data.mnist.MNIST_DATASET_fedml import get_mnist_dataloaders as get_mnist_fedml_dataloaders
# mnist_ld
from data.mnist.MNIST_LD import get_mnist_dataloaders as get_mnist_ld_dataloaders

# flickr
from data.flickr.FLICKR_DATASET import get_flickr_dataloaders
# celeba
from data.celeba.CELEBA_DATASET import get_celeba_dataloaders
# har
from data.har.HAR_DATASET import get_har_dataloaders


class SERVER_BASE(BASE):
    def __init__(self,
                 algorithm='fedavg',
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
                 alpha=0.5,
                 note=''):
        BASE.__init__(self, algorithm=algorithm,
                      seed=seed,
                      epoch=epoch,
                      model_name=model_name,
                      dataset_name=dataset_name,
                      lr=lr,
                      batch_size=batch_size,
                      lr_decay=lr_decay,
                      decay_step=decay_step)

        self.params = self.model.state_dict()
        self.updates = []
        self.selected_clients = []
        self.clients_per_round = clients_per_round
        self.rounds = rounds
        self.eval_interval = eval_interval
        self.alpha = alpha
        self.note = note

        self.optim = {'round': 0, 'acc': -1.0, 'params': None, 'loss': 10e8}

        self.train_writer = SummaryWriter(
            f'../../visualization/{self.algorithm}/{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{batch_size}_lr{lr}_train_{note}')
        self.test_writer = SummaryWriter(
            f'../../visualization/{self.algorithm}/{dataset_name}_{model_name}_C{clients_per_round}_E{epoch}_B{batch_size}_lr{lr}_val_{note}')
        self.Client = self.select_client()

        self.clients = self.setup_clients()
        assert self.clients_per_round <= len(self.clients)

        self.surrogates = self.setup_surrogates()
        assert len(self.surrogates) == clients_per_round

    def select_client(self):
        Client = None
        if self.algorithm == 'fedavg':
            Client = FedAVG_Client
        elif self.algorithm == 'fedprox':
            Client = FedPROX_Client
        elif self.algorithm == 'fedper':
            Client = FedPER_Client
        elif self.algorithm == 'fedlg':
            Client = FedLG_Client
        elif self.algorithm == 'fedsp':
            Client = FedSP_Client
        elif self.algorithm == 'fedmc':
            Client = FedMC_Client
        elif self.algorithm == 'fedmc-wo':
            Client = FedMC_WO_Client
        elif self.algorithm == 'fedmoe':
            Client = FedMoE_Client
        else:
            print("unimplemented algorithm")
            exit(0)
        return Client

    def setup_datasets(self):
        users = []
        trainloaders, testloaders = [], []
        if self.dataset_name == 'cifar10':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            users, trainloaders, testloaders = get_cifar10_dataloaders(batch_size=self.batch_size,
                                                                       train_transform=train_transform,
                                                                       test_transform=test_transform)
        elif self.dataset_name == 'cifar10_dirichlet':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            users, trainloaders, testloaders = get_cifar10_dirichlet_dataloaders(users_num=30, alpha=self.alpha,
                                                                                 batch_size=self.batch_size,
                                                                                 train_transform=train_transform,
                                                                                 test_transform=test_transform)
        elif self.dataset_name == 'cifar10_ld':
            users, trainloaders, testloaders = get_cifar10_ld_dataloaders()
        elif self.dataset_name == 'cifar100':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            users, trainloaders, testloaders = get_cifar100_dataloaders(batch_size=self.batch_size,
                                                                        train_transform=train_transform,
                                                                        test_transform=test_transform)
        elif self.dataset_name == 'cifar100_superclass':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])
            users, trainloaders, testloaders = get_cifar100_superclass_dataloaders(batch_size=self.batch_size,
                                                                                   train_transform=train_transform,
                                                                                   test_transform=test_transform)

        elif self.dataset_name == 'mnist_malposition':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            users, trainloaders, testloaders = get_mnist_malposition_dataloaders(batch_size=self.batch_size,
                                                                                 train_transform=train_transform,
                                                                                 test_transform=test_transform)
        elif self.dataset_name == 'mnist_wo_malposition':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            users, trainloaders, testloaders = get_mnist_wo_malposition_dataloaders(batch_size=self.batch_size,
                                                                                    train_transform=train_transform,
                                                                                    test_transform=test_transform)

        elif self.dataset_name == 'mnist_ld':
            users, trainloaders, testloaders = get_mnist_ld_dataloaders(batch_size=self.batch_size)
        elif self.dataset_name == 'mnist_fedml':
            users, trainloaders, testloaders = get_mnist_fedml_dataloaders(batch_size=self.batch_size)
        elif self.dataset_name == 'femnist':
            users, trainloaders, testloaders = get_femnist_dataloaders(batch_size=self.batch_size)
        elif self.dataset_name == 'flickr':
            users, trainloaders, testloaders = get_flickr_dataloaders(split_ratio=0.9, batch_size=10)
        elif self.dataset_name == 'celeba':
            users, trainloaders, testloaders = get_celeba_dataloaders(batch_size=self.batch_size)
        elif self.dataset_name == 'har':
            users, trainloaders, testloaders = get_har_dataloaders(batch_size=self.batch_size)
        return users, trainloaders, testloaders

    def setup_clients(self):
        users, trainloaders, testloaders = self.setup_datasets()
        clients = [
            self.Client(user_id=user_id,
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
            self.Client(user_id=i,
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

        for (client_samples, client_params) in updates:
            total_weight += client_samples

        (client_samples, new_params) = copy.deepcopy(updates[0])

        for k in new_params.keys():
            new_params[k] *= float(client_samples / total_weight)

        for k in new_params.keys():
            for i in range(1, len(updates)):
                client_samples, client_params = updates[i]
                w = client_samples / total_weight
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
                # if i >= 68 and self.selected_clients[k].user_id == 3:
                #     print("client3, skip")
                #     pass
                # print(self.selected_clients[k].user_id)
                surrogate = self.surrogates[k]
                c = self.selected_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_shared_params(self.params)
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
            surrogate.set_shared_params(self.params)
            num_test_samples, acc, loss = surrogate.test(dataset=dataset)
            acc_list.append((num_test_samples, acc))
            loss_list.append((num_test_samples, loss))
        return acc_list, loss_list

    def save_model(self):
        path = f'../../result/{self.algorithm}/{self.dataset_name}_{self.model_name}_C{self.clients_per_round}_E{self.epoch}_B{self.batch_size}_lr{self.lr}_{self.note}'
        if not os.path.exists(path):
            os.makedirs(path)
        path = f'{path}/model.pkl'
        print(f"model saved to：{path}")
        torch.save(self.params, path)
