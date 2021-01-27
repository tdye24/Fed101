import os
import sys
from models.utils.args import parse_args
from models.utils.model_utils import Logger
from models.server import Server


if __name__ == '__main__':
    sys.stdout = Logger()

    args = parse_args()
    algorithm = args.algorithm
    dataset = args.dataset
    partitioning = args.partitioning
    model_name = args.model
    pretrain_model = args.pretrain_model
    epoch = args.epoch
    num_rounds = args.num_rounds
    eval_interval = args.eval_interval
    all_clients_num = args.all_clients_num
    clients_per_round = args.clients_per_round
    batch_size = args.batch_size
    mini_batch = args.mini_batch
    seed = args.seed
    lr = args.lr
    lr_decay = args.lr_decay
    decay_step = args.decay_step

    print(f"#############Running {algorithm} With#############\n"
          f"algorithm:\t\t\t{algorithm}\n"
          f"dataset:\t\t\t{dataset}\n"
          f"partitioning:\t\t{partitioning}\n"
          f"all clients:\t\t{all_clients_num}\n"
          f"clients:\t\t\t{clients_per_round}\n"
          f"model name:\t\t\t{model_name}\n"
          f"pre-trained model:\t{pretrain_model}\n"
          f"epochs:\t\t\t\t{epoch}\n"
          f"num rounds:\t\t\t{num_rounds}\n"
          f"eval interval:\t\t{eval_interval}\n"
          f"batch size:\t\t\t{batch_size}\n"
          f"mini batch:\t\t\t{mini_batch}\n"
          f"lr:\t\t\t\t\t{lr}\n"
          f"lr-decay:\t\t\t{lr_decay}\n"
          f"decay_step:\t\t\t{decay_step}")

    server = Server(algorithm=algorithm,
                    rounds=num_rounds,
                    epoch=epoch,
                    pretrain_model=pretrain_model,
                    all_clients_num=all_clients_num,
                    clients_per_round=clients_per_round,
                    eval_interval=eval_interval,
                    seed=seed,
                    dataset_name=dataset,
                    partitioning=partitioning,
                    model_name=model_name,
                    lr=lr,
                    batch_size=batch_size,
                    mini_batch=mini_batch,
                    lr_decay=lr_decay,
                    decay_step=decay_step)
    server.initiate()
    server.federate()
    server.print_optim()
    # server.client_info(user_index=0)
