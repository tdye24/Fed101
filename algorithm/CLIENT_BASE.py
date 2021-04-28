import torch
import numpy as np
import torch.optim as optim
from utils.model_utils import batch_data

from algorithm.BASE import BASE


class CLIENT_BASE(BASE):
    def __init__(self, algorithm, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        BASE.__init__(self, algorithm=algorithm, seed=seed, epoch=epoch, model_name=model_name,
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
                              weight_decay=1e-4)

        batch_loss = []
        for epoch in range(self.epoch):
            for step, (data, labels) in enumerate(self.trainloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                batch_loss.append(loss.item())

        num_train_samples, update = self.trainloader.sampler.num_samples, self.get_params()
        # 异常检测
        if np.isnan(sum(batch_loss) / len(batch_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

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

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def get_shared_params(self):
        return self.model.cpu().base.state_dict()

    def set_shared_params(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key.startswith('shared'):
                tmp_params[key] = value
        self.set_params(tmp_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainloader = client.trainloader
        self.testloader = client.testloader
