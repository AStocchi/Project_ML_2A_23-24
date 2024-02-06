import os
import json
from collections import defaultdict

import torch
import random
import matplotlib.pyplot as plt

import numpy as np
from torchvision.models import resnet18

import datasets.ss_transforms as sstr
import datasets.np_transforms as nptr

from torch import nn
from client import Client
from rotate_client import Client_rot_dataset
from SR_client import Client_SReg
from ClientScaffold import clientScaffold
from datasets.femnist import Femnist
from server import Server
from ServerScaffold import serverScaffold
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

from models.convnn import Convnn
from models.features_net import Features_net


'''
#flag di comando del codice:

-niid (o iid)

-selection: 
    -> unif, 
    -> bias, 
    -> dyn, 
    -> poc
  %%-uniform:
  %%-dynamic:

-aggregation: 
    -> avg, 
    -> fedavg, 
    -> 'accur', 
    -> scaffold         fa passare a client di tipo scaffold

-domain_gen:  
  # flag per alg. di domain_generalization 
    -> rotate               esegue selezione client e divisione in gruppi E RUOTA 
    -> rotate_S_Regul   esegue selezione client e divisione in gruppi E RUOTA + LOSS regularized
    -> n                esegue normale client Fed
  %%-algorithm

'''


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset_num_classes(dataset):
    if dataset == 'idda':
        return 16
    if dataset == 'femnist':
        return 62
    raise NotImplementedError


def set_metrics(args):
    num_classes = get_dataset_num_classes(args.dataset)
    if args.model == 'deeplabv3_mobilenetv2':
        metrics = {
            'eval_train': StreamSegMetrics(num_classes, 'eval_train'),
            'test_same_dom': StreamSegMetrics(num_classes, 'test_same_dom'),
            'test_diff_dom': StreamSegMetrics(num_classes, 'test_diff_dom')
        }
    elif args.model == 'resnet18' or args.model == 'Convnn':
        metrics = {
            'eval_train': StreamClsMetrics(num_classes, 'eval_train'),
            'test': StreamClsMetrics(num_classes, 'test')
        }
    else:
        raise NotImplementedError
    return metrics


def model_init(args):
    if args.model == 'deeplabv3_mobilenetv2':
        return deeplabv3_mobilenetv2(num_classes=get_dataset_num_classes(args.dataset))
    if args.model == 'resnet18':
        model = resnet18()
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=get_dataset_num_classes(args.dataset))
        return model
    if args.model == 'Convnn':
        if args.domain_gen == 'rotate_S_Regul':
            model = Features_net()  # creiamo modello per setting di representation learning (domain gen. + regularized)
            return model
        model = Convnn()
        return model
    raise WrongModelError


def get_transforms(args):
    # TODO: test your data augmentation by changing the transforms here!
    if args.model == 'deeplabv3_mobilenetv2':
        train_transforms = sstr.Compose([
            sstr.RandomResizedCrop((512, 928), scale=(0.5, 2.0)),
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = sstr.Compose([
            sstr.ToTensor(),
            sstr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.model == 'Convnn' or args.model == 'resnet18':
        train_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
        test_transforms = nptr.Compose([
            nptr.ToTensor(),
            nptr.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise NotImplementedError
    return train_transforms, test_transforms


# funzione che carica porzione del dataset in memoria (INTERO, su colab la memoria non basta!)
def read_femnist_dir(data_dir, num, ds_load):
    data = defaultdict(lambda: {})
    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    i = 0  # questo permette di caricare in memoria solamente il dataset in posizione 'num'
    if num < 0 or num >= 36:
        print("Wrong Number of dataset Blocks")
    if ds_load == "multiple":
        for f in files:
            if i <= num:  # la condizione in if i<k, serve per prendere solo k file
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                data.update(cdata['user_data'])
            i = i + 1
    elif ds_load == "single":  # abbiamo usato questo metodo per fare il training centralizzato sul dataset 'sezionato'
        for f in files:
            if i == num:  # la condizione in if i==k, serve per prendere solo 1 file
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
                data.update(cdata['user_data'])
            i = i + 1

    return data


def read_femnist_data(train_data_dir, test_data_dir, num, ds_load):
    return read_femnist_dir(train_data_dir, num, ds_load), read_femnist_dir(test_data_dir, num, ds_load)


def get_datasets(args, num, ds_load="null"):
    if ds_load == "null":
        print("Ds_load variable lack of argument")

    train_datasets = []
    train_transforms, test_transforms = get_transforms(args)

    if args.dataset == 'idda':
        root = 'data/idda'
        with open(os.path.join(root, 'train.json'), 'r') as f:
            all_data = json.load(f)
        for client_id in all_data.keys():
            train_datasets.append(IDDADataset(root=root, list_samples=all_data[client_id], transform=train_transforms,
                                              client_name=client_id))
        with open(os.path.join(root, 'test_same_dom.txt'), 'r') as f:
            test_same_dom_data = f.read().splitlines()
            test_same_dom_dataset = IDDADataset(root=root, list_samples=test_same_dom_data, transform=test_transforms,
                                                client_name='test_same_dom')
        with open(os.path.join(root, 'test_diff_dom.txt'), 'r') as f:
            test_diff_dom_data = f.read().splitlines()
            test_diff_dom_dataset = IDDADataset(root=root, list_samples=test_diff_dom_data, transform=test_transforms,
                                                client_name='test_diff_dom')
        test_datasets = [test_same_dom_dataset, test_diff_dom_dataset]

    elif args.dataset == 'femnist':

        train_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if args.niid else 'iid', 'train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'niid' if args.niid else 'iid', 'test')

        train_data, test_data = read_femnist_data(train_data_dir, test_data_dir, num, ds_load)

        train_transforms, test_transforms = get_transforms(args)

        train_datasets, test_datasets = [], []

        for user, data in train_data.items():
            train_datasets.append(Femnist(data, train_transforms, user))
        for user, data in test_data.items():
            test_datasets.append(Femnist(data, test_transforms, user))
    else:
        raise NotImplementedError

    return train_datasets, test_datasets


def generate_base_clients(args, train_datasets, test_datasets, model):

    clients = [[], []]
    if args.domain_gen == 'n':
        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Client(args, ds, model, test_client=(i == 1)))
    elif args.domain_gen == 'rotate':
        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Client_rot_dataset(args, ds, model, test_client=(i == 1)))
    elif args.domain_gen == 'rotate_S_Regul':
        for i, datasets in enumerate([train_datasets, test_datasets]):
            for ds in datasets:
                clients[i].append(Client_SReg(args, ds, model, test_client=(i == 1)))
    else:
        raise NotImplementedError

    if args.aggregation_mode == 'scaffold':
        zero_list = []
        for i, k in enumerate(clients[0][0].model.parameters()):
            zero_list.append(torch.zeros_like(k))
        train_clients = []
        test_clients = []
        for i, _ in enumerate(clients[0]):
            train_clients.append(clientScaffold(clients[0][i], zero_list))
            test_clients.append(clientScaffold(clients[1][i], zero_list))

        return train_clients, test_clients
    else:
        return clients[0], clients[1]



def generate_dom_gen_clients(args, model):

    dataset_load = "single"
   
    dg_clients = []
    tot_dg_clients = 0
    dg_clients_train = []
    dg_clients_test = []

    num_datas = 0
    # selezione e salvataggio dei client
    while num_datas <= 11:  # ciclo che scorre sui dataset, per non caricarli tutti in memoria. Seleziona e estrae i client

        # print('Generate datasets...')
        train_datasets, test_datasets = get_datasets(args, num_datas, dataset_load)
        # print('Done.')

        print('Generate Dom-gen clients... ' + str(num_datas))
        train_clients, test_clients = generate_base_clients(args, train_datasets, test_datasets, model)
        tr_te_clients = list(zip(train_clients, test_clients))
        num_sel_clients = min(84, 1002 - tot_dg_clients)  # prendiamo 28 clients da ogni blocco e quelli che rimangono (22) dall'ultimo  (ci sono 36 blocchi)
        dg_select = np.random.choice(np.arange(len(tr_te_clients)), num_sel_clients, replace=False)

        for i in list(dg_select):
            dg_clients.append(tr_te_clients[i])
        tot_dg_clients += len(dg_select)
        num_datas += 1

    # divisione in gruppi
    group_div = np.random.choice(np.arange(len(dg_clients)), (6, 167), replace=False)

    for k, i in enumerate(group_div):
        for j in i:
            dg_clients[j][0].group = k
            dg_clients[j][1].group = k

    dg_select_train = [x for x, _ in dg_clients]
    dg_select_test = [y for _, y in dg_clients]
    print("Dataset/Clients - DOMAIN GEN. - Completed")

    return dg_select_train, dg_select_test


def gen_clients(args, model):

    if args.domain_gen == 'rotate' or args.domain_gen == 'rotate_S_Regul':
        train_clients, test_clients = generate_dom_gen_clients(args, model) 
    else:
        # caso normale (dati non ruotati)
        print('Generate datasets...')
        dataset_load = "multiple"
        quanti_blocchi_dataset = 12  # variabile che definisce quanti blocchi di dataset importare in memoria: [int]-> valori da 1 a 36  #tipicamente 5 (in tal caso vengono presi i dataset: 0,1,2,3,4)
        train_datasets, test_datasets = get_datasets(args, quanti_blocchi_dataset - 1, dataset_load)
        print('Done.')
        train_clients, test_clients = generate_base_clients(args, train_datasets, test_datasets, model)

    return train_clients, test_clients



def gen_server(args, train_clients, test_clients, model, metrics):

    if args.aggregation_mode == 'scaffold':
        zero_list = []
        for i, k in enumerate(train_clients[0].client.model.parameters()):
            zero_list.append(torch.zeros_like(k))
        server = serverScaffold(args, train_clients, test_clients, model, metrics, zero_list)
    else:
        server = Server(args, train_clients, test_clients, model, metrics)

    return server







#####################################################################################

# codice per effettivi esperimenti computazionali in ambito FEDERATED: server + client
def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    print('Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    metrics = set_metrics(args)

  
    train_clients, test_clients = gen_clients(args, model)
    num_clients = len(train_clients)
    print('clients ok  -  num_cl: ' + str(num_clients))

    # training federato unico (a prescindere da come sono stati generati i clients)

    server = gen_server(args, train_clients, test_clients, model, metrics)
    print('server ok')

    fed_model = server.train()
    test_val = server.test()
    print(fed_model)
    print(test_val)  
    
    
    # 0,88 test accuracy con 2 round e 10 epoche
    # 0,88  ''      ''   con 10 r. e 5 epoche
    # controllare a cosa serve la selezione randomica dei client (ad oggni round i modelli non sono tutti uguali?) -> risposta: Sì, ma cosa è diverso sono i dataset a disposizione dei vari client (cosa fondamentale soprattutto nel caso di distrib. NON-IID)
    # capire perchè batch size 64? è la dimensione perchè ci sono 64 caratteri? sono le direzioni di ottimizzazione?

    print("Federated END")
    '''
    vengono poste delle probabilità di selezione biased
    10% client -> prob. .5
    30% client -> prob. .0001

    -> queste sono da considerare probabilità aggregate di selezionare quel gruppo e poi all' interno di ognuno si sceglie uniformemente?
    -> questo porterebbe ad avere un gruppo di 60% -> prob. .4999 -> circa il 70% starebbe a prob. 1, però suddiviso 10%->p=.5 e 60%->p=.5 e poi uniformemente all'interno di ognuno
    '''


if __name__ == '__main__':
    main()