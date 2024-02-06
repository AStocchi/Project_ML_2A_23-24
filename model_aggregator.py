import copy
from collections import OrderedDict

import numpy as np
import math
import torch
import matplotlib as plt


class Model_aggregator:

    def __init__(self, args, train_clients, test_clients):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.vect_models = []


    def FedAvg_aggregation(self, updates, clients):
        aggregated_model = copy.deepcopy(updates[0])

        for param_idx, param in enumerate(aggregated_model.parameters()):
            weighted_params = torch.zeros_like(param.data)
            num_samples = 0
            samples = []
            for cl in clients:
                samples.append(np.floor(len(cl.dataset) / self.args.bs) * self.args.bs)
            num_samples = sum(samples)
            for i, model in enumerate(updates):
                for k, val in enumerate(model.parameters()):
                    if k == param_idx:
                        weighted_params += val * samples[i] / num_samples

            aggregated_param = weighted_params

            param.data = aggregated_param

        return aggregated_model


    def Avg_aggregation(self, updates):
        aggregated_model = copy.deepcopy(updates[0])

        for param_idx, param in enumerate(aggregated_model.parameters()):
            weighted_params = torch.zeros_like(param.data)
            num_clients = len(updates)
            for model in updates:
                for k, val in enumerate(model.parameters()):
                    if k == param_idx:
                        weighted_params += val * 1 / num_clients

            aggregated_param = weighted_params

            param.data = aggregated_param

        return aggregated_model


    def Accuracy_Aggregation(self, updates, clients):   #da finire, capire come ottenere accuracy scambiando i modelli!!
        aggregated_model = copy.deepcopy(updates[0])

        for param_idx, param in enumerate(aggregated_model.parameters()):
            weighted_params = torch.zeros_like(param.data)
            accuracy = 0
            acc_vect = []
            for cl in clients:
                acc_vect = 1 #mettere roba cl.cur_accuracy
            accuracy = sum(acc_vect)
            for i, model in enumerate(updates):
                for k, val in enumerate(model.parameters()):
                    if k == param_idx:
                        weighted_params += val * acc_vect[i] / accuracy

            aggregated_param = weighted_params

            param.data = aggregated_param

        return aggregated_model


    def aggregate(self, updates, clients):  # V1 V2 SCAFFOLD
        """
            This method handles the FedAvg aggregation
            :param updates: updates received from the clients
            :return: aggregated parameters
        """
        # FedAvg
        if self.args.aggregation_mode == 'FedAvg':
            aggregated_model = self.FedAvg_aggregation(updates, clients)

        # di seguito aggregation pesata con l'accuracy
        elif self.args.aggregation_mode == 'accuracy':
            aggregated_model = self.Accuracy_Aggregation(updates, clients)

        # aggregation con media normale
        elif self.args.aggregation_mode == 'average':
            aggregated_model = self.Avg_aggregation(updates)
        else:
            raise NotImplementedError
        return aggregated_model
