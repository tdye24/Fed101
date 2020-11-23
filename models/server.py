import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from models.utils.model_utils import read_data
# noinspection PyUnresolvedReferences
from tensorboardX import SummaryWriter
from models.client import Client
plt.switch_backend('agg')


i_ = 0
while os.path.exists('../visualization/fed' + str(i_)):
    i_ += 1
writer = SummaryWriter('../visualization/fed' + str(i_))


class Server:
    def __init__(self, model_path, seed=123, rounds=20, epoch=1, clients_per_round=1, eval_interval=1, dataset_name='femnist', model_name='cnn', lr=3e-4, batch_size=1, mini_batch=0.1):
        self.seed = seed  # randomly sampling
        self.clients = self.setup_clients(dataset_name, model_name=model_name, lr=lr, batch_size=batch_size, mini_batch=mini_batch)
        self.model_path = model_path
        self.rounds = rounds
        self.epoch = epoch
        self.clients_per_round = min(clients_per_round, len(self.clients))
        if len(self.clients) > 0:
            self.params = self.clients[0].model.state_dict()
        else:
            print("Error：clients长度为0")
            exit(0)
        self.eval_interval = eval_interval
        self.updates = []
        self.acc_over_sub = []  # 在选中的clients上的准确率
        self.acc_over_all = []  # 在所有clients上的准确率
        self.selected_clients = []
        self.optim = {'round': 0, 'acc': 0.0, 'params': None}  # 第几轮，准确率，最高准确率对应的参数

    # setup clients
    @staticmethod
    def setup_clients(dataset_name, model_name: str, batch_size: int, mini_batch: float, lr: float):
        train_data_dir = os.path.join('..', 'data', dataset_name, 'data', 'train')
        test_data_dir = os.path.join('..', 'data', dataset_name, 'data', 'test')

        users, train_data, test_data = read_data(train_data_dir, test_data_dir)
        clients = [
            Client(user_id, train_data[user_id], test_data[user_id], model_name=model_name, batch_size=batch_size,
                   mini_batch=mini_batch, lr=lr) for user_id in users]
        return clients

    def select_clients(self, round_th):
        num_clients = min(self.clients_per_round, len(self.clients))
        np.random.seed(seed=round_th)
        selected_clients = np.random.choice(self.clients, num_clients, replace=False)
        self.selected_clients = selected_clients
        # return selected_clients

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
        for i in range(self.rounds):
            # 该条语句不能往后放，因为接下来要测试模型在选中的clients上的性能
            self.select_clients(round_th=i)
            if i % self.eval_interval == 0:
                # 按理来说应该只传给被选中的clients，
                # 但是此处是为了测试global model在所有的clients上的平均准确率
                for c in self.clients:
                    c.set_params(self.params)
                # 测试当前模型在选中clients和所有clients上的准确率
                print("Round {}".format(i), end=' ')
                acc_over_all = self.test(domain="all")
                self.acc_over_all = acc_over_all
                acc_over_sub = self.test(domain="sub")
                self.acc_over_sub = acc_over_sub

                avg_acc_all = self.avg_metric(acc_over_all)
                avg_acc_sub = self.avg_metric(acc_over_sub)
                # 决定是否保存当前模型为最优模型
                if avg_acc_all > self.optim['acc']:
                    print("***当前最优模型***SAVE***")
                    self.optim.update({'round': i, 'acc': avg_acc_all, 'params': self.params})
                    self.save_model()

                # visualization
                writer.add_scalar('fedavg_all', avg_acc_all, global_step=i)
                writer.add_scalar('fedavg_sub', avg_acc_sub, global_step=i)

            # clients训练one by one
            for c in self.selected_clients:
                # TODO 之前server测试的时候将所有的client model设置为global model
                #  此处重复设置client model（仅仅是实验需求）
                # 每次下发最新模型
                c.set_params(self.params)
                # 每次下发在所有clients上的最优模型
                # c.set_params(self.optim['params'])
                # 在选中的clients上继续进行训练，并传回参数
                num_train_samples, update = c.train()
                self.updates.append((num_train_samples, update))

            # 联邦平均
            self.average()

            # 清空本轮状态
            self.updates = []
            self.acc_over_all = []
            self.acc_over_sub = []

    def test(self, domain: str):
        """
        返回所有clients的平均准确率
        :param domain: 参数是all,平均self.clients,参数是sub,平均self.selected_clients
        :return: 准确率列表，格式[（num_samples, acc）,...]
        """
        if domain == "all":
            for c in self.clients:
                num_test_samples, acc = c.test()
                self.acc_over_all.append((num_test_samples, acc))
            res_list = self.acc_over_all
            self.acc_over_all = []
            print("Average accuracy of all the clients: {}".format(self.avg_metric(res_list)))
        elif domain == "sub":
            for c in self.selected_clients:
                num_test_samples, acc = c.test()
                c.acc_list.append(acc)  # client维护自己训练过程中的accuracy list
                self.acc_over_sub.append((num_test_samples, acc))
            res_list = self.acc_over_sub
            self.acc_over_sub = []
            print("\t\tAverage accuracy of the selected clients: {}".format(self.avg_metric(res_list)))
        return res_list

    def print_optim(self):
        for c in self.clients:
            c.set_params(self.params)  # 下发最终的模型
        print("Round {}".format(self.rounds), end=' ')
        acc_all = self.test(domain="all")
        acc_sub = self.test(domain="sub")
        avg_acc_all = self.avg_metric(acc_all)
        avg_acc_sub = self.avg_metric(acc_sub)

        if avg_acc_all > self.optim['acc']:
            print("***当前最优模型***SAVE***")
            self.optim.update({'round': self.rounds, 'acc': avg_acc_all, 'params': self.params})
            self.save_model()
        print(
            f"######Round: {self.optim['round']}, Optimal Federated Model, Average Accuracy Over All Clients(AAAC): {self.optim['acc']}######")

    def client_info(self, user_index):
        for i in range(len(self.clients[user_index].loss_list)):
            writer.add_scalar('client {} loss'.format(user_index), self.clients[user_index].loss_list[i], global_step=i)
        for i in range(len(self.clients[user_index].acc_list)):
            writer.add_scalar('client {} acc'.format(user_index), self.clients[user_index].acc_list[i], global_step=i)

    def save_model(self):
        torch.save(self.params, self.model_path)