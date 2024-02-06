import copy
from collections import OrderedDict

import numpy as np
import math
import torch
import matplotlib as plt
from server import Server


class serverScaffold(Server):

    def __init__(self, args, train_clients, test_clients, model, metrics, zero_list):
        super(serverScaffold, self).__init__(args, train_clients, test_clients, model, metrics)
        self.x = copy.deepcopy(model)  # x è il modello globale 
        self.c = copy.deepcopy(zero_list)  # c è la direzione di update globale (cioè del server)
        self.eta_g = 1


    # metodo per iterare su parameters
    def param_iter(self, model, el_index, b_index):
        if type(model) == list:
            tmp = model
        else:
            tmp = model.parameters()
        for e_i, elem in enumerate(tmp):
            for b_i, block in enumerate(elem):
                if e_i == el_index and b_i == b_index:
                    return block

    # metodo per contare gli elementi di model
    def param_count(self, model):
        dim_elem_model = []
        count_b = 0
        for e_i, elem in enumerate(model.parameters()):
            for b_i, block in enumerate(elem):
                count_b += 1
            dim_elem_model.append(count_b)
            count_b = 0
        return dim_elem_model
		
		
	# metodo scaffold che gestisce update del modello sul server e del vettore direzione c globale
    def scaffold(self, updates, clients):   
        num_params = self.param_count(self.x)
        for n_el, el in enumerate(num_params):  #scorre elementi del modello
            for n in np.arange(el):   #scorre parametri dell'elemento
                for i, ml in enumerate(updates):   #per ogni clients esegue aggiornamento del parametro
                    py = self.param_iter(ml, n_el, n)
                    px = self.param_iter(self.x, n_el, n)
                    px.data += self.eta_g * 1 / len(clients) * (px.data - py.data)
                    #edit n_el linea sotto (prima era el)
                    pci = self.param_iter(clients[i].ci, n_el, n)
                    new_pci = self.param_iter(clients[i].new_ci, n_el, n)
                    pc = self.param_iter(self.c, n_el, n)
                    pc += 1 / len(self.train_clients) * (new_pci - pci)


    def aggregate(self, updates, clients):  #override method
        """
            This method handles the FedAvg aggregation
            :param updates: updates received from the clients
            :return: aggregated parameters
        """
        aggregated_model = self.scaffold(updates, clients)

        return aggregated_model
		
		
    def train_round(self, clients):  #override method
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        global_model = copy.deepcopy(clients[0].client.model)
        for i, cl in enumerate(clients):
            # print('train_client : ' + str(i))
            cl.model = global_model
            cl.train(self.c)  #aggiornato richiamo del client.train(), visto lavoriamo con un clientScaffold
            updates.append(cl.pass_model())
        return updates
