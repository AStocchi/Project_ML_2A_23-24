import copy
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


from scipy import ndimage
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction

class clientScaffold():
    def __init__(self, client, zero_list):
        self.client = client
        self.eta_l = 1
        self.ci = copy.deepcopy(zero_list)   #liste inizializzate a 0, con "struttura degli elementi"  simile a client.model
        self.new_ci = copy.deepcopy(zero_list) 
        self.pkl_ci = "ci_" + str(self.client) + ".pkl"
        self.pkl_client_model = "client_model_" + str(self.client) + ".pkl" 
        self.dump_model(self.ci, self.pkl_ci)
        self.dump_model(self.client.model, self.pkl_client_model)
        self.ci = -1
        self.new_ci = -1


    def __str__(self):
        return self.client.name 

    def load_model(self, file_name):
        new_path = "pklfile/" + file_name
        with open(new_path, 'rb') as pickle_file:
          return pkl.load(pickle_file, fix_imports=True, encoding='ASCII', errors='strict', buffers=None)

    def dump_model(self, model, file_name):
        #aggiunta del path della cartella al nome del file per il salvataggio del modello
        new_path = "pklfile/" + file_name
        with open(new_path, 'wb') as pickle_file:
            pkl.dump(model, pickle_file, protocol=None, fix_imports=True, buffer_callback=None)
            


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

  # 30/12/2023: nelle prossime due funzioni mettiamo self.client.model al posto di self.model 

	# metodo scaffold che gestisce update del modello sui client
    def scaffold_on_clients(self, c_server):
        num_params = self.param_count(self.client.model)
        for n_el, el in enumerate(num_params):  # scorre elementi del modello
            for n in np.arange(el):	 #scorre i parameters
                py = self.param_iter(self.client.model, n_el, n)
                # inserire load del modello proprio qui
                # dove fare il dump?
                  # fare load/dump al richiamo/rilascio del client
                pci = self.param_iter(self.ci, n_el, n)
                pc = self.param_iter(c_server, n_el, n)
                py.data -= self.eta_l * (pc.data - pci)

	# metodo per calcolare il vettore c da usare alla prossima iterazione (c = grad_locale(model_glob) -> c = variazione del modello locale dopo il primo update della prima epoca? O NO???)
    def compute_new_ci(self, cent_model):          # cent_model --> centralized model
        num_params = self.param_count(self.client.model)
        for n_el, el in enumerate(num_params): 
            for nb in np.arange(el):
                py = self.param_iter(self.client.model, n_el, nb)
                px = self.param_iter(cent_model, n_el, nb)
                # load/dump modello proprio qui?
                # proviamo a farlo solo a inzio e fine ottimizz, altrimenti troppo tempo
                pci = self.param_iter(self.new_ci, n_el, nb)
                pci = px.data - py.data
                #print(n_el, nn, py, px, pci)  #print di prova


    def run_epoch(self, cur_epoch, optimizer, cost_function, c_server):  #metodo "override" manuale per scaffold
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        samples = 0
        cumulative_loss = 0
        cumulative_accuracy = 0
        self.client.model.train()

        for cur_step, (images, labels) in enumerate(self.client.train_loader):
            cuda_device = torch.device("cuda:0")

            # if cur_step%1000==0:
            #    print("  -epoch: " + str(cur_epoch) + "  -iteration: " + str(cur_step))

            # Forward pass
            y = torch.stack(images, dim=1)  # y[1] è immagine trasformata da vettore a matrice
            value_bs = y.size()
            
            z = self.client.transform_image(y, value_bs[0])

            z = z.float()
            z = z.to(cuda_device)
			
            # Reset the optimizer (spostato, va fatto prima del resto [in realtà probabilmente non cambia nulla])
            optimizer.zero_grad()
            outputs, loss = self.client.output_and_loss(z, labels, cost_function, cuda_device)

				#comando aggiunto in SCAFFOLD
            if cur_epoch == 0:
                cent_model = copy.deepcopy(self.client.model)
				
				
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

				#comando aggiunto in SCAFFOLD
            if cur_epoch == 0:
              self.compute_new_ci(cent_model)

				
				#comando aggiunto in SCAFFOLD
            self.scaffold_on_clients(c_server)
				

            # Better print something, no?
            samples += z.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            predicted = predicted.to('cpu')
            cumulative_accuracy += predicted.eq(labels).sum().item()

        return cumulative_loss / samples, cumulative_accuracy / samples * 100



    def train(self, c_server=None):  #metodo "override" manuale per scaffold
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        cost_function = self.client.get_cost_function()

        optimizer = self.client.get_optimizer(self.client.args.lr, self.client.args.wd, self.client.args.m)  # TO DO fare tuning dei parametri

        ### load del modello da fare qui
        self.ci = self.load_model(self.pkl_ci)
        self.new_ci = copy.deepcopy(self.ci)
        self.client.model = self.load_model(self.pkl_client_model)

        for epoch in range(self.client.args.num_epochs):
            _, _ = self.run_epoch(epoch, optimizer, cost_function, c_server)

        self.ci = self.new_ci  #una volta per ogni train (cioè per ogni round) aggiorno vettore c locale della direzione SCAFFOLD
        #self.dump_model(self.ci) giusto farlo qui? No



    def test(self, metric):  #metodo "override" manuale per scaffold
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        if self.client.model == -1:
            self.client.model = self.load_model(self.pkl_client_model)   #carico il modello dal file pickle
        self.client.test(metric)
        self.client.model = -1



    def pass_model(self):
        tmp = copy.deepcopy(self.client.model)
        self.dump_model(self.ci, self.pkl_ci)
        self.dump_model(self.client.model, self.pkl_client_model)
        self.pkl_ci = -1
        self.client.model = -1
        return tmp
