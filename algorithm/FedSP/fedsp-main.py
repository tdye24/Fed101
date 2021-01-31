import sys
from utils.args import parse_args
from utils.model_utils import Logger
from algorithm.FedSP.server import Server


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
    note = args.note

    print(f"#############Running FedSP With#############\n"
          f"algorithm:\t\t\tFedSP\n"
          f"dataset:\t\t\t{dataset}\n"
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
          f"decay_step:\t\t\t{decay_step}\n"
          f"note:\t\t\t\t{note}")

    server = Server(rounds=num_rounds,
                    epoch=epoch,
                    pretrain_model=pretrain_model,
                    clients_per_round=clients_per_round,
                    eval_interval=eval_interval,
                    seed=seed,
                    dataset_name=dataset,
                    model_name=model_name,
                    lr=lr,
                    batch_size=batch_size,
                    mini_batch=mini_batch,
                    lr_decay=lr_decay,
                    decay_step=decay_step,
                    note=note)
    server.initiate()
    server.federate()
    server.print_optim()
