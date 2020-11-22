import torch
import torch.nn.functional as F
import torch.optim as optim
from models.femnist.cnn import Net
from models.utils.model_utils import batch_data


class Client:
    def __init__(self, user_id, train_data, test_data, model: str, lr=3e-4, batch_size=10, mini_batch=None, epoch=1, seed=123):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.select_model(model)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.seed = seed
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mini_batch = mini_batch

    # TODO (self.model移到init函数)
    def select_model(self, model):
        if model == 'cnn':
            model = Net()
            self.model = model.to(self.device)

    def train(self):
        self.model.train()
        # batch大小，可以指定batch_size，也可以指定mini_batch
        if self.mini_batch is None:
            num_data = self.batch_size
        else:
            frac = min(1.0, self.mini_batch)
            num_data = max(1, int(frac * len(self.train_data['y'])))
        BEST_ACC = 0
        for i in range(self.epoch):
            for batched_x, batched_y in batch_data(self.train_data, num_data, self.seed):
                input_data, target_data = self.process_data(batched_x, batched_y)
                self.optimizer.zero_grad()
                output = self.model(input_data)
                loss = self.criterion(output, target_data)
                loss.backward()
                self.optimizer.step()
                acc = self.test()
                if acc > BEST_ACC:
                    BEST_ACC = acc
        # print("user: {}, acc: {}, best acc: {}".format(self.user_id, acc, BEST_ACC))
        num_train_samples, update = len(self.train_data['y']), self.get_params()
        return num_train_samples, update, acc

    def test(self):
        self.model.eval()
        test_data_x = torch.tensor(self.test_data['x']).to(self.device)
        test_data_y = torch.tensor(self.test_data['y']).to(self.device)
        output = self.model(test_data_x)
        output = torch.argmax(output, dim=-1)
        acc = torch.sum(output == test_data_y).float() / len(test_data_y)
        return acc

    def process_data(self, raw_x, raw_y):
        input_data, target_data = torch.tensor(raw_x).to(device=self.device), torch.tensor(raw_y).to(device=self.device)
        return input_data, target_data

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)
