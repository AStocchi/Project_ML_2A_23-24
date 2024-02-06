import copy
import torch
import matplotlib.pyplot as plt

from scipy import ndimage
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader
from rotate_client import Client_rot_dataset

from utils.utils import HardNegativeMining, MeanReduction

class Client_SReg(Client_rot_dataset):
    def __init__(self, args, dataset, model, test_client=False, group=0):
        super(Client_SReg, self).__init__(args, dataset, model, test_client, group)

        self.model = model
        self.L2R_coeff = 0.01  # taken from github repo
        self.CMI_coeff = 0.001  # taken from github repo
        self.n_feat = 1024
        self.r_mu = nn.Parameter(torch.zeros(62, self.n_feat, requires_grad=True), requires_grad=True)
        self.r_sigma = nn.Parameter(torch.ones(62, self.n_feat, requires_grad=True), requires_grad=True)
        self.C = nn.Parameter(torch.ones([], requires_grad=True), requires_grad=True)

    def _get_outputs(self, images): #override _get_outputs method
        '''
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        if self.args.model == 'Convnn':
            return self.model(images)  # self.model(images) e self.model.forward(images) sono scritture analoghe, danno lo stesso risultato
        '''
        
        #fa effetivamente quello che vogliomo tolgo l'if perchè fare come commentato sopra non serve
        # estraggo prima le feature, poi costruisco le distribuzioni e campiono
        z_featurized = self.model.featurize(images)  # z_featurized sono le medie e le varianze
        z_samples = self.model.get_sample(z_featurized)
        # passo i sample al classificatore
        output = self.model.classify(z_samples)
        return output, z_featurized


    def compute_loss(self, output, labels, cost_function, cuda_device, z_params): #ovveride compute_loss method
        # regL2R = torch.zeros_like(loss)
        # regCMI = torch.zeros_like(loss)

        # calcolo la norm2 delle feature (media, var), sarà utilizzata come termine di loss ulteriore per la regolarizzazione
        regL2R = z_params.norm(dim=1).mean()  # funzione che calcola la norma2 dei parametri estratti dalla rete model
        # per ogni set di valori (1 vettore di medie/var. per ogni immagine nel mini-batch) calcola la norma
        # poi fa la media su tutti i vettori ( =>ha peso invariante rispetto alla batch size )

        # computo il valore del termine loss relativo alle feature "intermedie", z_med e z_var   #utilizzando la divergenza kullback-lebovitz  -> sort of distanza tra distribuzioni
        z_mu = z_params[:, :self.n_feat]
        z_sigma = nn.functional.softplus(z_params[:, self.n_feat:])
        r_sigma_softplus = nn.functional.softplus(self.r_sigma)
        r_mi = self.r_mu[labels]
        r_mi = r_mi.to(cuda_device)
        r_sig = r_sigma_softplus[labels]
        r_sig = r_sig.to(cuda_device)
        z_mu_scaled = z_mu * self.C
        z_sigma_scaled = z_sigma * self.C

        regCMI = torch.log(r_sig) - torch.log(z_sigma_scaled) + (z_sigma_scaled ** 2 + (z_mu_scaled - r_mi) ** 2) / (
                    2 * r_sig ** 2) - 0.5
        regCMI = regCMI.sum(1).mean()
        # definendo prima i pesi che hanno nella loss
        reg_loss = self.L2R_coeff * regL2R + self.CMI_coeff * regCMI



        labels = labels.to(cuda_device)
        loss = cost_function(output, labels)
        loss = loss + reg_loss
        return loss


    def get_optimizer(self, lr, wd, momentum): #ovveride get_optimizer method
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        #if not (self.r_mu.is_leaf and self.r_sigma.is_leaf and self.C.is_leaf):
            #breakpoint()
        optimizer.add_param_group({'params': [self.r_mu, self.r_sigma, self.C], 'lr': lr, 'momentum': momentum})
        return optimizer

    def only_output(self, z): #override only_output method
        outputs, _ = self._get_outputs(z)
        return outputs

    def output_and_loss(self, z, labels, cost_function, cuda_device): #override output_and_loss method
        outputs, z_featurized = self._get_outputs(z)
        # Apply the loss
        loss = self.compute_loss(outputs, labels, cost_function, cuda_device, z_featurized)
        # su output e z_featurized calcoliamo le loss
        # come minchia funziona il passaggio di back-propag. (visto che i modelli sono separati in fase di costruzione?)
        # come posso trainare l'uno e l'altro? posso farlo contemporaneamente? è tutto automatizzato?
        # -> copiare da sergio&tizi paper -> usare passo back() normale!
        return outputs, loss




