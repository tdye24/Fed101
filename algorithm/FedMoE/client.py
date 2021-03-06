import torch
import torch.optim as optim

from algorithm.BASE import BASE


class Client(BASE):
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        BASE.__init__(self, algorithm='fedmoe', seed=seed, epoch=epoch, model_name=model_name,
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

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(params=model.parameters(), lr=self.lr * self.lr_decay ** (round_th / self.decay_step))

        batch_loss = []
        for epoch in range(self.epoch):
            for step, (data, labels) in enumerate(self.trainloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                output, avg_gate_output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
        num_train_samples, global_feature_update, fc_update = self.trainloader.sampler.num_samples, self.get_global_feature_params(), self.get_fc_params()
        return num_train_samples, global_feature_update, fc_update, sum(batch_loss) / len(batch_loss)

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
            gate_global = []
            gate_local = []

            for step, (data, labels) in enumerate(dataloader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                output, avg_gate_output = model(data)
                loss = criterion(output, labels)
                output = torch.argmax(output, dim=-1)
                total_right += torch.sum(output == labels)
                total_samples += len(labels)

                avg_gate_output = avg_gate_output.cpu().numpy()
                gate_global.append(avg_gate_output[0])
                gate_local.append(avg_gate_output[1])

            acc = float(total_right) / total_samples
            avg_gate_global = sum(gate_global) / len(gate_global)
            avg_gate_local = sum(gate_local) / len(gate_local)

        return total_samples, acc, loss.item(), avg_gate_global, avg_gate_local

    def get_params(self):
        return self.model.cpu().state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def get_global_feature_params(self):
        return self.model.cpu().global_feature.state_dict()

    def set_global_feature_params(self, global_feature_params):
        self.model.global_feature.load_state_dict(global_feature_params)

    def get_fc_params(self):
        return self.model.cpu().fc.state_dict()

    def set_fc_params(self, fc_params):
        self.model.fc.load_state_dict(fc_params)

    def update(self, client):
        self.model.load_state_dict(client.model.state_dict())
        self.trainloader = client.trainloader
        self.testloader = client.testloader
