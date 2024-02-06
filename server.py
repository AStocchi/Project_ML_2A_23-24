import copy
from collections import OrderedDict
from model_aggregator import Model_aggregator
from client_selector_server import Client_selector

import numpy as np
import math
import torch
import matplotlib as plt


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.selection = args.selection  
        self.aggregation_mode = args.aggregation_mode #--> FedAvg, accuracy

        self.client_selector = Client_selector(args, train_clients, test_clients)
        self.model_aggregator = Model_aggregator(args, train_clients, test_clients)




    def aggregate(self, updates, clients): 
        """
            This method handles the FedAvg aggregation
            :param updates: updates received from the clients
            :return: aggregated parameters
        """
        aggregated_model = self.model_aggregator.aggregate(updates, clients)
        return aggregated_model


    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        global_model = copy.deepcopy(clients[0].model)
        for i, clien in enumerate(clients):
            # print('train_client : ' + str(i))
            clien.model = global_model
            clien.train()
            updates.append(clien.model)

        return updates


    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """

        for r in range(self.args.num_rounds):
            print("  -round:" + str(r))

            [clients, clients_indexes] =  self.client_selector.select_clients(self.metrics)

            updates = self.train_round(clients)

            print("aggregation global model")
            FedAvg = self.aggregate(updates, clients)  #creazione del nuovo modello globale

            for tr_cl in self.train_clients:
                tr_cl.model = FedAvg  ###passare anche al test client?  teoricamente NON SERVE! avviene automaticamente

            print("evaluation")
            e_val = self.eval_train(clients)
            print(e_val)

            if type(clients_indexes) == list:  # da usare SOLO per il modo dinamico (*), aggiornamento delle probabilità
                self.client_selector.prob_assignment(e_val, clients_indexes, r)  

            #if r == 9 or r == 49 or r == 99 or r == 249 or r == 499 or r == 999:
            if r == 499:
                for te_cl in self.test_clients:
                    te_cl.model = FedAvg
                r_test = self.test()
                f = open('record_file.txt', 'a')
                f.write(str(r_test))
                f.write('\n')
                f.close()
        return FedAvg


    def eval_train(self, clients_to_evaluate=[]):
        """
        This method handles the evaluation on the train clients
        """
        eval_vect_metrics = []

        if not len(clients_to_evaluate):
            # se non si passa alcuna lista l'evaluation viene eseguita su tutti i client
            for client in self.train_clients:
                self.metrics['eval_train'].reset()
                client.test(self.metrics['eval_train'])
                self.metrics['eval_train'].get_results()
                eval_vect_metrics.append(self.metrics['eval_train'].results['Overall Acc'])
        else:
            for client in clients_to_evaluate:
                self.metrics['eval_train'].reset()
                client.test(self.metrics['eval_train'])
                self.metrics['eval_train'].get_results()
                eval_vect_metrics.append(self.metrics['eval_train'].results['Overall Acc'])

        return eval_vect_metrics


    def test(self):
        """
            This method handles the test on the test clients
        """
        test_vect_metrics = []

        for client in self.test_clients:
            self.metrics['test'].reset()
            client.test(self.metrics['test'])
            self.metrics['test'].get_results()
            test_vect_metrics.append(self.metrics['test'].results['Overall Acc'])

        return test_vect_metrics
