import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as distributions

from models.convnn import Convnn



class Features_net(Convnn):
    def __init__(self):
        super(Features_net, self).__init__()
        #cambiamo struttura rete 
        #n_feat definisce la dimensione dello spazio (potenzialmente espanso) dei dati -> (rappresenta la quantità di caratteristiche utili a descrivere l'oggetto) n_feat può essere >> di n_classi 
                #-> visto che verrà richiesta la norma2 del vettore medie/var. , alzare troppo la dimensionalità (n_feat alto) porta ad un effetto di diluizione: tante feature -> norma2 del vettore + alta -> alta loss (invece poche feature, a parità di loss, hanno maggior possibilità di variare)
                    
        #self.fc1 = nn.Linear(2048, n_feat*2)   #cambio ultimo layer -> ora fa estrazione delle features [medie, varianze]
        #self.last_layer = nn.Linear(n_feat, 62)   #definisco nuovo layer che farà da classificatore

        #variamo solamente num. neuroni per rendere utilizzabile la rete nel nuovo setting
        self.fc1 = nn.Linear(1024, 62)
        

    def featurize(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # flatten the feature maps into a long vector
        out = self.fc(out)
        return out


    def get_sample(self, z_params):
        
        num_samples = 64  #n_samples definisce il numero di sample estratti dalla distribuzione (1 sample è un vettore di dim=n_feat)
        n_feat = 1024
        z_mu = z_params[:, :n_feat]
        z_sigma = F.softplus(z_params[:, n_feat:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
        z_sample = z_dist.rsample()#.view([n_feat])
        return z_sample


    def classify(self, z_sample):
        out = self.fc1(z_sample)
        return out


