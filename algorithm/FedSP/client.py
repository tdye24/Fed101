import torch
import numpy as np
import torch.optim as optim
from utils.model_utils import batch_data

from algorithm.BASE import BASE


class Client(BASE):
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        BASE.__init__(self, algorithm='fedsp', seed=seed, epoch=epoch, model_name=model_name,
                      lr=lr, lr_decay=lr_decay, decay_step=decay_step)

        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader

        self.loss_list = []
        self.acc_list = []

    def train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(params=model.parameters(),
                              lr=self.lr * self.lr_decay ** (round_th / self.decay_step),
                              weight_decay=1e-3)

        batch_loss = []
        # for epoch in range(self.epoch):
        #     for step, (data, labels) in enumerate(self.trainloader):
        #         data = data.to(self.device)
        #         labels = labels.to(self.device)
        #         optimizer.zero_grad()
        #         output = model(data)
        #         loss = criterion(output, labels)
        #         loss.backward()
        #         optimizer.step()
        #         batch_loss.append(loss.item())
        for epoch in range(self.epoch):
            for batched_x, batched_y in batch_data(self.trainloader, self.batch_size, seed=self.seed):
                input_data = self.process_x(batched_x)
                target_data = self.process_y(batched_y)
                input_data = torch.tensor(input_data).cuda()
                input_data = input_data.float()
                target_data = torch.tensor(target_data).cuda()
                target_data = target_data.long()
                optimizer.zero_grad()
                output = model(input_data)
                loss = criterion(output, target_data)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

        # num_train_samples, update = self.trainloader.sampler.num_samples, self.get_global_feature_params()
        num_train_samples, update = len(self.trainloader['y']), self.get_global_feature_params()
        return num_train_samples, update, sum(batch_loss) / len(batch_loss)

    def test(self, dataset='test'):
        model = self.model
        model.eval()
        model.to(self.device)

        criterion = torch.nn.CrossEntropyLoss()

        if dataset == 'test':
            dataloader = self.testloader
        else:
            dataloader = self.trainloader

        batch_loss = []
        total_right = 0
        total_samples = 0
        # with torch.no_grad():
        #     for step, (data, labels) in enumerate(dataloader):
        #         data = data.to(self.device)
        #         labels = labels.to(self.device)
        #         output = model(data)
        #         loss = criterion(output, labels)
        #         output = torch.argmax(output, dim=-1)
        #         total_right += torch.sum(output == labels)
        #         total_samples += len(labels)
        with torch.no_grad():
            input_data = self.process_x(dataloader['x'])
            target_data = self.process_y(dataloader['y'])
            input_data = torch.tensor(input_data).cuda()
            input_data = input_data.float()
            target_data = torch.tensor(target_data).cuda()
            target_data = target_data.long()
            output = model(input_data)
            loss = criterion(output, target_data)
            output = torch.argmax(output, dim=-1)
            total_right = torch.sum(output == target_data)
            total_samples = len(target_data)
            batch_loss.append(loss.item())
            acc = float(total_right) / total_samples

        return total_samples, acc, loss.item()

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def get_global_feature_params(self):
        return self.model.cpu().global_feature.state_dict()

    def set_global_feature_params(self, global_feature_params):
        self.model.global_feature.load_state_dict(global_feature_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainloader = client.trainloader
        self.testloader = client.testloader

    @staticmethod
    def process_x(raw_x_batch):
        return np.array(raw_x_batch)

    @staticmethod
    def process_y(raw_y_batch):
        return np.array(raw_y_batch)
