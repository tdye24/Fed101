import os
import copy
import torch
from torchvision import transforms
import numpy as np

from algorithm.BASE import BASE
from algorithm.SERVER_BASE import SERVER_BASE


class Server(SERVER_BASE):
    def __init__(self,
                 algorithm='fedmc_wgan',
                 seed=123,
                 rounds=20,
                 epoch=1,
                 clients_per_round=1,
                 eval_interval=1,
                 dataset_name='femnist',
                 model_name='femnist',
                 lr=3e-4,
                 batch_size=1,
                 lr_decay=0.99,
                 decay_step=200,
                 alpha=0.5,
                 note=''):
        SERVER_BASE.__init__(self,
                             algorithm=algorithm,
                             seed=seed,
                             rounds=rounds,
                             epoch=epoch,
                             clients_per_round=clients_per_round,
                             eval_interval=eval_interval,
                             dataset_name=dataset_name,
                             model_name=model_name,
                             lr=lr,
                             batch_size=batch_size,
                             lr_decay=lr_decay,
                             decay_step=decay_step,
                             alpha=alpha,
                             note=note)
        self.train_clients = None  # for meta training
        self.test_clients = None  # for meta test

    # rewrite federate function
    def federate(self):
        print("Begin Federating!")
        print(f"Training {len(self.clients)} clients!")
        for i in range(self.rounds):
            self.select_clients(round_th=i)

            self.train_clients = self.selected_clients[0: self.clients_per_round // 2]
            self.test_clients = self.selected_clients[self.clients_per_round // 2:]

            # meta train
            for k in range(len(self.train_clients)):
                surrogate = self.surrogates[k]
                c = self.train_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_g_encoder_critic(self.params)
                num_train_samples, update, loss = surrogate.meta_train(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((num_train_samples, copy.deepcopy(update)))

            # average
            self.average()

            # meta test
            for k in range(len(self.test_clients)):
                surrogate = self.surrogates[k]
                c = self.test_clients[k]
                # surrogate <-- c
                surrogate.update(c)
                surrogate.set_g_encoder_critic(self.params)
                num_train_samples, update, loss = surrogate.meta_test(round_th=i)
                # c <-- surrogate
                c.update(surrogate)
                self.updates.append((num_train_samples, copy.deepcopy(update)))

            # average
            self.average()

            # clear
            self.updates = []
            self.selected_clients = []

            if i % self.eval_interval == 0:
                print("--------------------------\n")
                print("Round {}".format(i))
                # test on training data
                acc_over_all, loss_over_all = self.test(dataset='train')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TRAIN# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                self.train_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.train_writer.add_scalar('loss', avg_loss_all, global_step=i)
                # test on testing data
                acc_over_all, loss_over_all = self.test(dataset='test')
                avg_acc_all, avg_loss_all = self.avg_metric(acc_over_all), self.avg_metric(loss_over_all)
                print("#TEST# Avg acc: {:.4f}%, Avg loss: {:.4f}".format(avg_acc_all * 100, avg_loss_all))

                if avg_acc_all > self.optim['acc']:
                    print("\033[1;31m" + "***Best Model***SAVE***" + "\033[0m")
                    self.optim.update({'round': i, 'acc': avg_acc_all, 'params': self.params, 'loss': avg_loss_all})
                    # self.save_model()

                self.test_writer.add_scalar('acc', avg_acc_all, global_step=i)
                self.test_writer.add_scalar('loss', avg_loss_all, global_step=i)
