import copy
from collections import OrderedDict

import numpy as np
import math
import torch
import matplotlib as plt
import operator 


class Client_selector:

    def __init__(self, args, train_clients, test_clients):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.leave_one_out()
        self.selection = args.selection # selez. statica:  # y -> selezione uniforme ; n -> selezione biased
        #self.dynamic = dynamic# y-> selez. dinamica ; n -> selez. statica ; PoC -> selez. PowerOfChoiche
        if self.selection != 'unif':
            self.prob = np.ones(len(self.train_clients)) / len(self.train_clients)
            if self.selection == 'dyn':
                self.ro = np.zeros(len(self.train_clients))  # utile per selez. dinamica
                self.round_vision = -np.ones(len(self.train_clients))  # utile per selez. dinamica
            elif self.selection == 'PoC':
                self.flag_PoC = 0


    def prob_assignment_Boltzman(self, eval_vect_metrics, clients_indexes, r):  # SELECTION dynamic (attenzione che deve essere richiamato fuori)
        z = 1  # da far variare per trovare valore ottimale
        never_seen = []
        # if self.round_vision[c] != -1:
        # print("già visto")
        for c, slave in enumerate(self.train_clients):
            if c in clients_indexes:  # se client è appena stato visto

                # ro = (accuracy => eval_vect_metrics[k] )/10*z   # k è la posizione, che il client numero 'c' ha, all'interno del vettore degli indici: clients_indexes
                self.ro[c] = eval_vect_metrics[np.where(clients_indexes == c)[0][0]] / 10 * z
                self.round_vision[c] = r
            elif self.round_vision[c] != -1:  # aggiornamento di ro ad ogni nuovo round
                # if self.ro[c] > 0:
                self.ro[c] -= 1 / 30 * z
                # if self.ro[c] < 0:   #quando l'esponente ritorna a 0 -> valore viene mantenuto per i round successivi e non superato
                #  self.ro[c] = 0   #in realtà potrebbe avere comunque un senso lasciarlo andare in negativo! (da testare entrambe)
            else:
                never_seen.append(c)  # se un client non è mai stato visto il suo indice viene salvato in una lista

        self.ro[never_seen] = (-(r ** 2) / 30) * z  # lista che viene aggiornata una volta sola

        self.prob = math.e ** (-self.ro)  # calcolo le 'prob'
        tot = sum(self.prob)
        self.prob = self.prob / tot  # normalizzo, ottenendo effettivamente delle probabilità

        # sarebbe da plottare la distribuzione di probabilità ad ogni giro (ordinata cresc. o descresc.)(per vedere come cambia)
        # if not r%2:
        # if not r%10 or r==65:
        # z=1
        # plt.pyplot.bar(np.arange(len(self.train_clients)),np.sort(self.prob))   #problema: il plot si chiude alla chiusura della funzione (cioè subito)
        return self.prob


    def dynamic_selection_clients(self):  # SELECTION dynamic
        '''
          selezione dei client in modo dinamico privilegiando quelli non visti poi quelli con bassa accuracy
          ma riportando tutte le
        '''
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        selected_clients = []

        indexes_selected_clients = np.random.choice(np.arange(len(self.train_clients)), num_clients, p=self.prob,replace=False)

        for i in indexes_selected_clients:
            selected_clients.append(self.train_clients[i])

        return selected_clients, indexes_selected_clients


    def assign_proportional_prob_static(self):  # SELECTION PoC (singleton, per definire prob.)
        self.prob = []
        for client in self.train_clients:
            self.prob.append(len(client.dataset))
        self.prob = np.array(self.prob)
        self.prob = self.prob / sum(self.prob)

    def power_of_choice_selection_clients(self, metrics):  # SELECTION PoC
        '''
          selezione dei client usando l'algoritmo PowerOfChoiche presentato in [17]
        '''
        C = 0.1  # frazione di client da valutare  #(se num_db= 5 ) c=0.2 -> d=10 (random sampling) ; c=0.4 -> d=20 ; c=1 -> d=50
        K = len(self.train_clients)  # client totali
        d = int(K * C)  # numero client da valutare -> size di A
        num_clients_TOtrain = min(self.args.clients_per_round, d)  # numero client da trainare -> size di S_t
        clients_loss = []
        selected_clients = []

        # self.assign_proportional_prob_static()    #fatta una sola volta nel train per velocizzare l'esecuzione # DA CORREGGERE FARE IN CLIENT_SELECTOR

        indexes_sampled_clients = np.random.choice(np.arange(len(self.train_clients)), d, p=self.prob, replace=False)
        for i in indexes_sampled_clients:
            metrics['test'].reset()
            self.train_clients[i].test(metrics['test'])
            metrics['test'].get_results()
            clients_loss.append(metrics['test'].results['Overall Acc'])
        list_poc = zip(clients_loss, indexes_sampled_clients)
        list_poc = sorted(list_poc)
        clients_loss = [x for x, _ in list_poc]
        indexes_sampled_clients = [y for _, y in list_poc]
        indexes_selected_clients = indexes_sampled_clients[-num_clients_TOtrain:]
        for i in indexes_selected_clients:
            selected_clients.append(self.train_clients[i])

        return selected_clients

    def static_select_clients(self):  # SELECTION uniform o biased
        '''
          se la probabilità con cui vengono selezionati i client non è uniforme dobbiamo runnare questa parte di codice
          quello che è possibile fare per generalizzare il tutto e passare questa informazione quando si crea il server
          'non_uniform' e 'uniform'
        '''
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        selected_clients = []

        if self.selection == 'bias':
            total_clients = len(self.train_clients)
            num_set_A = int(np.floor(total_clients * 0.1))
            num_set_B = int(np.floor(total_clients * 0.3))
            num_set_C = total_clients - num_set_A - num_set_B

            set_A = self.train_clients[: num_set_A]
            set_B = self.train_clients[num_set_A: (num_set_A + num_set_B)]
            set_C = self.train_clients[-num_set_C:]

            selected_index = np.floor(np.random.rand(num_clients) * (
                    10 ** 5))  # selezionamo un certo num_clients di slave da usare un questo round
            indexes = [0, 0, 0]
            for i in selected_index:
                if i < 5 * (10 ** 4):
                    indexes[0] += 1

                elif i >= 5 * (10 ** 4) + 10:
                    indexes[2] += 1

                else:
                    indexes[1] += 1
            selected_a = np.random.choice(set_A, indexes[0],replace=False)
            selected_b = np.random.choice(set_B, indexes[1], replace=False)
            selected_c = np.random.choice(set_C, indexes[2], replace=False)
            for sel in selected_a:
                selected_clients.append(sel)  # all'interno di ogni gruppo estraiamo uniformemente
            for sel in selected_b:
                selected_clients.append(sel)
            for sel in selected_c:
                selected_clients.append(sel)
        elif self.selection == 'unif':
            selected_clients = np.random.choice(self.train_clients, num_clients, replace=False)
        else:
            raise NotImplementedSelection

        return selected_clients

    

    def select_clients(self, metric):
        if self.selection == 'dyn':
            [clients, clients_indexes] = self.dynamic_selection_clients()  # modo dinamico (*)
            return clients, clients_indexes
        elif self.selection == 'PoC':
            if self.flag_PoC == 0:  # struttura di esecuzione Singleton dell'assegnazione delle probabilità
                self.assign_proportional_prob_static()
                self.flag_PoC = 1
            clients = self.power_of_choice_selection_clients(metric)
            clients_indexes = None
            return clients, clients_indexes
        else:
            clients = self.static_select_clients()  # da utilizzare quando i client vengono scelti in modo statico: uniform o biased
            clients_indexes = None
            return clients, clients_indexes

        # print(str(clients))  #controllo dei selezionati


    def leave_one_out(self):
        if self.args.leave_one_out != -1 and (self.args.domain_gen == 'rotate' or self.args.domain_gen == 'rotate_S_Regul'):
            c = 0
            while c < len(self.train_clients):
                if self.train_clients[c].group == self.args.leave_one_out:
                    self.train_clients.pop(c)
                    c = c - 1
                if c == len(self.train_clients)-1:
                    break
                c = c + 1

            c = 0
            while c < len(self.test_clients):
                if self.test_clients[c].group != self.args.leave_one_out:
                    self.test_clients.pop(c)
                    c = c - 1
                if c == len(self.test_clients)-1:
                    break
                c = c + 1


        
    