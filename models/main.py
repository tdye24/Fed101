import os
from models.utils.model_utils import read_data
from models.server import Server
from models.client import Client


# setup clients
def setup_clients(dataset, model=None):
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', 'test')

    users, train_data, test_data = read_data(train_data_dir, test_data_dir)
    clients_ = [Client(user_id, train_data[user_id], test_data[user_id], model, mini_batch=0.1) for user_id in users]
    return clients_


if __name__ == '__main__':
    clients = setup_clients('femnist', model='cnn')
    server = Server(clients, rounds=5000, epoch=1, clients_per_round=20, eval_interval=20, model_path='./femnist/femnist.pkl')
    server.federate()
    server.clients_accuracies()
    server.client_info(user_index=0)
