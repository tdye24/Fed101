import copy
import torch
import torch.optim as optim

from algorithm.BASE import BASE


class Client(BASE):
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        BASE.__init__(self, algorithm='fedprox', seed=seed, epoch=epoch, model_name=model_name,
                      lr=lr, lr_decay=lr_decay, decay_step=decay_step)

        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader

        self.loss_list = []
        self.acc_list = []

        self.start_point = None

    @staticmethod
    def model_difference(start_point, new_point):
        loss = 0
        old_params = start_point.state_dict()
        for name, param in new_point.named_parameters():
            loss += torch.norm(old_params[name] - param, 2)
        return loss

    def train(self, round_th):
        model = self.model
        model.to(self.device)

        self.start_point = copy.deepcopy(model)

        model.train()

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr * self.lr_decay ** (round_th / self.decay_step))

        batch_loss = []
        for epoch in range(self.epoch):
            for step, (data, labels) in enumerate(self.trainloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                difference = self.model_difference(self.start_point, model)
                loss = criterion(output, labels) + 0.1 / 2 * difference
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        num_train_samples, update = self.trainloader.sampler.num_samples, self.get_params()
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

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainloader = client.trainloader
        self.testloader = client.testloader
