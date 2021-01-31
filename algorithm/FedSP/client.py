import torch
import torch.optim as optim

# models
from models.fedsp.cifar10.CIFAR10 import CIFAR10
from models.fedsp.femnist.FEMNIST import FEMNIST
from models.fedsp.mnist.MNIST import MNIST


class Client:
    def __init__(self, user_id, trainloader, testloader, model_name: str, lr=3e-4, batch_size=10, mini_batch=-1, epoch=1,
                 seed=123, lr_decay=0.99, decay_step=200):
        torch.manual_seed(seed)  # recurrence experiment
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = batch_size
        self.model = None  # initialize
        self.select_model(model_name)

        self.lr = lr
        self.lr_decay = lr_decay
        self.decay_step = decay_step

        self.epoch = epoch
        self.mini_batch = mini_batch

        self.loss_list = []
        self.acc_list = []

        self.samples_num = trainloader.sampler.num_samples

    def select_model(self, model_name):
        model = None
        if model_name == 'femnist':
            model = FEMNIST()
        elif model_name == 'cifar10':
            model = CIFAR10()
        elif model_name == 'mnistcnn':
            model = MNIST()
        else:
            print("Unimplemented Model!")
            exit(0)
        self.model = model.to(self.device)

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
                output = model(data)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                # last epoch: batch_loss
                if epoch == self.epoch - 1:
                    batch_loss.append(loss.item())
        num_train_samples, update = self.samples_num, self.get_global_params()
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
        return self.model.cpu().state_dict()

    def get_global_params(self):
        return self.model.cpu().global_feature.state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)

    def set_global_params(self, model_params):
        self.model.global_feature.load_state_dict(model_params)