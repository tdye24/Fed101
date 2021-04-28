import torch
import numpy as np
import torch.optim as optim

from algorithm.BASE import BASE
from algorithm.CLIENT_BASE import CLIENT_BASE


class Client(CLIENT_BASE):
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        CLIENT_BASE.__init__(self,
                             algorithm='fedmc',
                             user_id=user_id,
                             trainloader=trainloader,
                             testloader=testloader,
                             seed=seed,
                             epoch=epoch,
                             model_name=model_name,
                             lr=lr,
                             lr_decay=lr_decay,
                             decay_step=decay_step)

    @staticmethod
    def w_distance(P, Q):
        batch_size = P.shape[0]
        length = P.shape[1]
        result = torch.zeros(batch_size).cuda()
        for i in range(batch_size):
            delta = torch.tensor(0).cuda()
            delta_lst = torch.zeros(length).cuda()
            for j in range(length):
                delta = delta + P[i][j] - Q[i][j]
                delta_lst[j] = delta
            torch.abs_(delta_lst)
            out = torch.sum(delta_lst)
            result[i] = out
        return torch.sum(result)

    def meta_train(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = False

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
                g_critic_out, l_critic_out, output = model(data)
                clf_loss = criterion(output, labels)
                w_distance = self.w_distance(g_critic_out, l_critic_out)
                loss = clf_loss - 3e-4 * w_distance
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                batch_loss.append(loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('critic'):
                param.requires_grad = True

        num_train_samples, update = self.trainloader.sampler.num_samples, self.get_params()

        # 异常检测
        if np.isnan(sum(batch_loss) / len(batch_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        return num_train_samples, update, sum(batch_loss) / len(batch_loss)

    def meta_test(self, round_th):
        model = self.model
        model.to(self.device)
        model.train()

        # frozen
        for (key, param) in model.named_parameters():
            if key.startswith('g'):
                param.requires_grad = False

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
                g_critic_out, l_critic_out, output = model(data)
                clf_loss = criterion(output, labels)
                w_distance = self.w_distance(g_critic_out, l_critic_out)
                loss = clf_loss - 3e-4 * w_distance
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                batch_loss.append(loss.item())

        # unfrozen
        for (key, param) in model.named_parameters():
            if key.startswith('g'):
                param.requires_grad = True

        num_train_samples, update = self.trainloader.sampler.num_samples, self.get_params()

        # 异常检测
        if np.isnan(sum(batch_loss) / len(batch_loss)):
            print(f"client {self.user_id}, loss NAN")
            exit(0)

        return num_train_samples, update, sum(batch_loss) / len(batch_loss)

    # rewrite
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
                g_critic2_out, l_critic2_out, output = model(data)
                loss = criterion(output, labels)
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)
            acc = float(total_right) / total_samples

        return total_samples, acc, loss.item()

    def set_g_encoder_critic(self, params):
        tmp_params = self.get_params()
        for (key, value) in params.items():
            if key.startswith('g') or key.startswith('critic'):
                tmp_params[key] = value
        self.set_params(tmp_params)
