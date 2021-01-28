import torch
import copy
import logging
import sys
import torch.nn.functional as F
import torch.optim as optim

# 模型
from models.femnist.femnist import FEMNIST
from models.cifar10.cifar10 import CIFAR10
from models.cifar10.mobilenet import mobilenet
from models.cifar100.cifar100 import CIFAR100
from models.synthetic_iid.fcnn import FCNN
from models.mnist.MNIST2NN import MNIST2NN
from models.mnist.MNISTCNN import MNISTCNN
from models.ml100k.ML100K import ML100K

from models.utils.model_utils import batch_data, compare_models
import inspect


# from models.utils.gpu_mem_track import MemTracker

# frame = inspect.currentframe()  # define a frame to track
# gpu_tracker = MemTracker(frame)  # define a GPU tracker


class Client:
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, batch_size=10, mini_batch=-1, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200, algorithm='fedavg'):
        torch.manual_seed(256)  # 复现实验，主要是模型初始化的点是一样的
        self.algorithm = algorithm
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = batch_size
        self.model = None  # 初始化
        self.select_model(model_name)
        # 不能定义为Client的一个属性，可能上一轮的信息，会影响下一轮的优化
        # self.optimizer = optim.SGD(params=self.model.parameters(), lr=lr)
        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.epoch = epoch
        self.seed = seed
        # self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.mini_batch = mini_batch
        self.loss_list = []
        self.acc_list = []
        # fedprox用
        self.start_point = None
        self.samples_num = len(trainloader.dataset._labels)

    def select_model(self, model_name):
        model = None
        if model_name == 'femnist':
            model = FEMNIST()
        elif model_name == 'cifar10':
            model = CIFAR10()
        elif model_name == 'cifar100':
            model = CIFAR100()
        elif model_name == 'fcnn':
            model = FCNN()
        elif model_name == 'mnist2nn':
            model = MNIST2NN()
        elif model_name == 'mnistcnn':
            model = MNISTCNN()
        elif model_name == 'mobile_net':
            model = mobilenet(alpha=1, class_num=10)
        elif model_name == 'ml100k':
            model = ML100K()
        else:
            print("Unimplemented Model!")
            exit(0)
        self.model = model.to(self.device)

    @staticmethod
    def model_difference(start_point, new_point):

        loss = 0
        old_params = start_point.state_dict()
        for name, param in new_point.named_parameters():
            # print(v-new_point[k])
            # res = (v == new_point[k]).flatten()
            # res = torch.sum(res) / res.shape[0]
            # print(res)
            # if name.endswith('.weight'):
            #     loss += torch.norm(old_params[name] - param, 2)
            # else:
            #     # name.endswith('.weight')
            #     pass

            loss += torch.norm(old_params[name] - param, 2)

        return loss

    def train(self, round_th):

        # round_th 该client参与的轮次
        # 测试
        # self.model.to(self.device)
        # self.select_model('mnist2nn')
        # self.model.load_state_dict(latest_params)

        # exit(0)
        # self.optimizer = optim.Adam(params=self.model.parameters(), lr=3e-4)
        # gpu_tracker.track()
        # self.model = self.model.to(self.device)
        # fedprox用
        # 必须要有deepcopy否则两个model params一起改变
        # https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html
        self.start_point = copy.deepcopy(self.model)
        # 我啥时候又重新定义了一个optimizer，惊了
        # self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)
        # TODO(ml)
        model = self.model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr * self.lr_decay ** (round_th / self.decay_step))

        # batch大小，可以指定batch_size，也可以指定mini_batch
        if self.mini_batch == -1:
            num_data = min(self.batch_size, self.samples_num)
        else:
            frac = min(1.0, self.mini_batch)
            num_data = max(1, int(frac * self.samples_num))
        batch_loss = []
        for epoch in range(self.epoch):
            # print(f"{self.user_id}, epoch {i} / {self.epoch}", self.model.state_dict())
            # 犯了一个错误，seed设置的是同一个值，导致训练数据shuffle没起作用，改成self.seed + i
            for step, (data, labels) in enumerate(self.trainloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = None
                if self.algorithm == 'fedavg':
                    loss = criterion(output, labels)
                elif self.algorithm == 'fedprox':
                    difference = self.model_difference(self.start_point, model)
                    loss = criterion(output, labels) + 0.5 * difference
                loss.backward()
                optimizer.step()
                # 最后一轮测一下batch_loss
                if epoch == self.epoch - 1:
                    batch_loss.append(loss.item())

            # for batched_x, batched_y in batch_data(self.train_data, num_data, self.seed + epoch + round_th):
            #     input_data, target_data = self.process_data(batched_x, batched_y)
            #     # if len(batched_y) <= 1:
            #     #     continue
            #     optimizer.zero_grad()
            #     output = model(input_data)
            #     loss = None
            #     if self.algorithm == 'fedavg':
            #         loss = criterion(output, target_data)
            #     elif self.algorithm == 'fedprox':
            #         difference = self.model_difference(self.start_point, model)
            #         loss = criterion(output, target_data) + 0.5 * difference
            #     loss.backward()
            #     optimizer.step()
            #     batch_loss.append(loss.item())
            # if len(batch_loss) > 0:
                # print(f"{self.user_id}. Local Training Epoch: {epoch} \tLoss: {sum(batch_loss) / len(batch_loss)}")

        num_train_samples, update = self.samples_num, self.get_params()

        # torch.cuda.empty_cache()
        # gpu_tracker.track()
        return num_train_samples, update, sum(batch_loss) / len(batch_loss)

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        if dataset == 'test':
            dataloader = self.testloader
        else:
            dataloader = self.trainloader

        total_right = 0
        total_samples = 0
        with torch.no_grad():
            for step, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output = model(data)
                loss = criterion(output, labels)
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, loss.item()

    def process_data(self, raw_x, raw_y):
        input_data, target_data = torch.tensor(raw_x).float().to(device=self.device), torch.tensor(raw_y).long().to(
            device=self.device)
        return input_data, target_data

    def get_params(self):
        # ml TODO(待验证)
        return self.model.cpu().state_dict()
        # return self.model.state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)
