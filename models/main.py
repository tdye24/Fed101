import os
from models.utils.args import parse_args

from models.server import Server


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    model_name = args.model
    model_path = args.model_path
    epoch = args.epoch
    num_rounds = args.num_rounds
    eval_interval = args.eval_interval
    clients_per_round = args.clients_per_round
    batch_size = args.batch_size
    mini_batch = args.mini_batch
    seed = args.seed
    lr = args.lr

    server = Server(rounds=num_rounds,
                    epoch=epoch,
                    clients_per_round=clients_per_round,
                    eval_interval=eval_interval,
                    model_path=model_path,
                    seed=seed,
                    dataset_name=dataset,
                    model_name=model_name,
                    lr=lr,
                    batch_size=batch_size,
                    mini_batch=mini_batch)
    server.federate()
    server.print_optim()
    server.client_info(user_index=0)
