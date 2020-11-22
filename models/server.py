import numpy as np
from collections import OrderedDict
from models.client import Client
from models.femnist.cnn import Net


class Server:
    def __init__(self, clients, seed=123, rounds=20, clients_per_round=1):
        self.seed = seed  # randomly sampling
        self.clients = clients
        self.rounds = rounds
        self.clients_per_round = clients_per_round
        if len(clients) > 0:
            self.params = clients[0].model.state_dict()
        else:
            print("Error：clients长度为0")
            exit(0)
        self.updates = []
        self.accuracies = []

    def select_clients(self, round_th):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed=round_th)
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        return selected_clients

    def average(self):
        total_weight = 0
        new_params = OrderedDict()
        for (client_samples, client_params) in self.updates:
            total_weight += client_samples
            for (k, v) in client_params.items():
                if k not in new_params:
                    new_params[k] = v * client_samples
                else:
                    new_params[k] += v * client_samples
        for k in new_params.keys():
            new_params[k] /= total_weight
        # 更新global模型参数
        self.params = new_params

        # 清空
        self.updates = []

    def avg_acc(self):
        total_weight = 0
        total_acc = 0
        for (client_samples, acc) in self.accuracies:
            total_weight += client_samples
            total_acc += client_samples * acc
        avg_acc = total_acc / total_weight
        # 清空
        self.accuracies = []
        return avg_acc

    def federate(self):
        print("Begin Federating!")
        for i in range(self.rounds):
            selected_clients = self.select_clients(round_th=i)
            for c in selected_clients:
                # p rint(c.get_params() == self.params)
                c.set_params(self.params)
                num_train_samples, update, acc = c.train()
                self.updates.append((num_train_samples, update))
                self.accuracies.append((num_train_samples, acc))
            self.average()
            if i % 20 == 0:
                print("Round {}, average accuracy: {}".format(i, self.avg_acc()))
