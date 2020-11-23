import torch
import torch.nn.functional as F
import torch.optim as optim
from models.femnist.cnn import Net
from models.femnist.resnet import ResNet, Basicblock
from models.utils.model_utils import batch_data


class Client:
    def __init__(self, user_id, train_data, test_data, model_name: str, lr=3e-4, batch_size=10, mini_batch=-1, epoch=1, seed=123):
        self.user_id = user_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.model = None  # 初始化
        self.select_model(model_name)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.seed = seed
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mini_batch = mini_batch
        self.loss_list = []
        self.acc_list = []

    # TODO (self.model移到init函数)
    def select_model(self, model_name):
        model = None
        if model_name == 'cnn':
            model = Net()
        elif model_name == 'resnet':
            model = ResNet(Basicblock, [1, 1, 1, 1], 62)
        else:
            print("Unimplemented Model!")
            exit(0)
        self.model = model.to(self.device)

    def train(self):
        self.model.train()
        # batch大小，可以指定batch_size，也可以指定mini_batch
        if self.mini_batch == -1:
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
                self.loss_list.append(loss)
                self.optimizer.step()
                # acc = self.test()
                # if acc > BEST_ACC:
                #     BEST_ACC = acc
        # print("user: {}, acc: {}, best acc: {}".format(self.user_id, acc, BEST_ACC))
        num_train_samples, update = len(self.train_data['y']), self.get_params()
        return num_train_samples, update

    def test(self):
        self.model.eval()
        test_data_x = torch.tensor(self.test_data['x']).to(self.device)
        test_data_y = torch.tensor(self.test_data['y']).to(self.device)
        output = self.model(test_data_x)
        output = torch.argmax(output, dim=-1)
        num_test_samples = len(self.test_data['y'])
        acc = torch.sum(output == test_data_y).float() / len(test_data_y)
        return num_test_samples, acc

    def process_data(self, raw_x, raw_y):
        input_data, target_data = torch.tensor(raw_x).to(device=self.device), torch.tensor(raw_y).to(device=self.device)
        return input_data, target_data

    def get_params(self):
        return self.model.state_dict()

    def set_params(self, model_params):
        self.model.load_state_dict(model_params)
