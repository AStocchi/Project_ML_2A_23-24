# import necessary libraries
import torch
import torchvision
from torchvision import transforms as T
import torch.nn.functional as F
import torch.nn as nn

class Convnn(torch.nn.Module):
    def __init__(self):
        super(Convnn,self).__init__()
        
        self.layer1 = nn.Sequential(
                #INIZIO: immagini 28x28 x1 channel
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=0),
                # Ogni canale di uscita del layer convoluzionale rappresenta un diverso filtro applicato all'immagine di partenza
                # -> 24x24 x32 ch
            #nn.BatchNorm2d(32),
            nn.ReLU(),
                # Le operazioni di pooling sono equivalenti all'utilizzo di un filtro convoluzionale, con la differenza che si
                # utilizza una funzione non lineare come Max con l'obiettivo di ridurre la dimensionalità dei dati
            nn.MaxPool2d(kernel_size=2, stride=2))
                # -> 12x12 x32 ch

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0),
                # -> 8x8 x64 ch
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
                # -> 4x4 x64 ch
                # l'output dell'ultimo filtro di Pooling è visto come il layer iniziale di una rete neurale "classica"
                # è quindi necessario trasformare vettorizzando 4x4x64 in 1024x1x1, ottenendo il "valore di ogni neurone di input"

        self.fc = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(1024, 2048),
            nn.ReLU())

        self.fc1 = nn.Sequential(
            #nn.Dropout(0.5),
            nn.Linear(2048, 62)) #ma qui ci mettiamo n_feat

        #layer tentativo di output 62 -> 1 tipo layer argmax (non da usare in realtà!!)
        #self.last = nn.Sequential( 
            #nn.Dropout(0.5),
            #nn.Linear(62, 1))


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) # flatten the feature maps into a long vector
        out = self.fc(out)
        out = self.fc1(out)
        return out
