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
from datasets.femnist import Femnist
from server import Server
from utils.args import get_parser
from datasets.idda import IDDADataset
from models.deeplabv3 import deeplabv3_mobilenetv2
from utils.stream_metrics import StreamSegMetrics, StreamClsMetrics

from models.convnn import Convnn
from models.features_net import Features_net



domain_gen = False  # flag per alg. di domain_generalization -> true esegue selezione client e divisione in gruppi -> false


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
            model = Features_net()  # creiamo modello per setting di representation learning
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

        train_data_dir = os.path.join('data', 'femnist', 'data', 'old_iid', 'old_train')
        test_data_dir = os.path.join('data', 'femnist', 'data', 'old_iid', 'old_test')

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


def gen_clients(args, train_datasets, test_datasets, model, dg=False):
    clients = [[], []]
    for i, datasets in enumerate([train_datasets, test_datasets]):
        for ds in datasets:
            clients[i].append(Client(args, ds, model, test_client=i == 1))
    return clients[0], clients[1]


def build_dom_gen_clients(args, model):
    dataset_load = "single"
    num_datas = 0

    dg_clients = []
    tot_dg_clients = 0

    dg_clients_train = []
    dg_clients_test = []

    # selezione e salvataggio dei client
    while num_datas <= 35:  # ciclo che scorre sui dataset, per non caricarli tutti in memoria. Seleziona e estrae i client

        # print('Generate datasets...')
        train_datasets, test_datasets = get_datasets(args, num_datas, dataset_load)
        # print('Done.')

        print('Generate clients...' + str(num_datas))
        train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, model)
        tr_te_clients = list(zip(train_clients, test_clients))
        num_sel_clients = min(28,
                              1002 - tot_dg_clients)  # prendiamo 28 clients da ogni blocco e quelli che rimangono (22) dall'ultimo  (ci sono 36 blocchi)
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

    dg_select_train = [x for x, _ in dg_clients]  # possiamo lavorare su dg_clients o necessita group_div??
    dg_select_test = [y for _, y in dg_clients]
    print("Dataset/Clients - DOMAIN GEN. - Completed")

    return dg_select_train, dg_select_test


#####################################################################################


def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    print('Initializing model...')
    model = model_init(args)
    model.cuda()
    print('Done.')

    metrics = set_metrics(args)


    ''' 
    #->unico dubbio era fare training tipo: blocco 1 -> 1000 epoche ==> blocco 2 -> 1000 epoche -> ...
    # da risultati identici a : 1° epoca = blocco 1 -> blocco 2 -> ...
                            # poi 2° epoca = blocco 1 -> ... ecc.
    # crediamo di sì, in quanto è solo un riordinamento dei dati che vengono visti
    # ma sappiamo che ogni blocco protrebbe potenzialmente deviare il modello verso una certa distribuzione, NON OTTENENDO una convergenza alla 'vera' distribuzione media
    '''

    if domain_gen:
        ##
        # ricordarsi di mettere args - epoch = 1   !!!!
        num_cycles = 20
        # args.num_epochs = 1   #serve a costruire client che eseguono train "autonomo" di 1 sola epoca, le epoche vengo tramutate in cicli esterni
        ####
        # ottieni i 1002 clients già selezionati e suddivisi
        train_clients, test_clients = build_dom_gen_clients(args, model)

        # training centralizzato su tutti i client
        print("Training DOM. GEN. clients")
        passing_model = model

        for cycle in np.arange(num_cycles):
            print(cycle)
            i = 0
            for tr in train_clients:
                if not i % 100:
                    print(i)
                tr.model = passing_model
                tr.train()
                passing_model = tr.model
                i = i + 1

        # fare test
        vect_metrics = []
        print('Testing models')

        i = 0
        for te in test_clients:
            if not i % 50:
                print(i)
                print(vect_metrics)
            metrics.reset()
            te.test(metrics['test'])
            metrics['test'].get_results()
            vect_metrics.append(metrics['test'].results['Overall Acc'])

            i += 1

    else:

        dataset_load = "single"

        vect_metrics = []

        for z in range(0,1,1):



                num_clients = 0

                # editare problema delle epoche consecutive

                print('Generate datasets...')
                train_datasets, test_datasets = get_datasets(args, num_clients, dataset_load)
                print('Done.')

                print('Generate swap-clients...')
                s_train_clients, s_test_clients = gen_clients(args, train_datasets, test_datasets, model)

                print("Train s-client: " + str(num_clients))
                s_train_clients[0].train()

                s_test_clients[0].test(metrics['test'])
                metrics['test'].get_results()
                vect_metrics.append(metrics['test'].results['Overall Acc'])

                # s_test_clients[0].model = s_train_clients[0].model #i modelli sono proprio gli stessi cioè
                # s_test_clients[0].test(metrics['test']) # gli passa quello già trainato, il passaggio esplicito non è necessario
                # metrics['test'].get_results()
                # vect_metrics.append(metrics['test'].results['Overall Acc'])

                num_clients = num_clients + 1


                print(vect_metrics)

                # ciclo che scorre sui dataset, facendo training con uno per volta, per non caricarli tutti in memoria
                while num_clients <= 11:

                    print('Generate datasets...')
                    train_datasets, test_datasets = get_datasets(args, num_clients,dataset_load)
                    print('Done.')

                    print('Generate clients...')
                    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, s_train_clients[
                        0].model)  # con modello parz. trainato passato da una "generazione" all'altra
                    # train_clients[0].model = s_train_clients[0].model

                    print("Train client: " + str(num_clients))
                    train_clients[0].train()

                    metrics['test'].reset()
                    test_clients[0].test(metrics['test'])
                    metrics['test'].get_results()
                    vect_metrics.append(metrics['test'].results['Overall Acc'])

                    s_train_clients[0].model = train_clients[0].model

                    if num_clients == 11 or num_clients == 35:

                        print(vect_metrics)

                    num_clients = num_clients + 1

                print("ciao")

                num_clients = 0
                metrics['test'].reset()
                while num_clients <= 11:

                    print('Generate datasets...')
                    train_datasets, test_datasets = get_datasets(args, num_clients,dataset_load)
                    print('Done.')

                    print('Generate clients...')
                    train_clients, test_clients = gen_clients(args, train_datasets, test_datasets, s_train_clients[
                        0].model)  # con modello parz. trainato passato da una "generazione" all'altra
                    # train_clients[0].model = s_train_clients[0].model

                    print("Train client: " + str(num_clients))
                    #train_clients[0].train()


                    test_clients[0].test(metrics['test'])
                    metrics['test'].get_results()
                    vect_metrics.append(metrics['test'].results['Overall Acc'])

                    s_train_clients[0].model = train_clients[0].model

                    if num_clients == 11 or num_clients == 35:

                        print(vect_metrics)

                    num_clients = num_clients + 1
                vect_metrics = []
        # domanda 1: ->RISOLTO: perchè le label e le immagini sono a 4 a 4? (valore di default del batch_size -> si trova in args.py)
        # domanda 2: output(images) perchè è sempre uguale anche se images sono diverse? (inoltre i 62 output sono score di similarità per le classi? o sono i pesi degli archi del 2° fully-connected layer?)
        # output cambiano solo al variare del numero di epoche di training -> un indizio a favore del fatto che siano gli archi!! -> male!! vorremmo fosse l'altro!
        # (in realtà è improbabile siano i valori degli archi, in training plottano output differenti => sensato, più probabile siano 62 output sono score di similarità per le classi)
        # OK, ma allora perchè sono sempre tutti uguali in test??? -> guarda sotto!
        # ->RISOLTO: batch size 4 fa cagare, intorno a 64 funziona bene, per raggiungere 89% aumentare num_epoc e dim dataset -> con dim = 1 e num_epoc = 15, lo superiamo addirittura 92%
        # domanda 3: se volessi salvare il modello come lo facciamo in modo che possa essere letto successivamente?

        # test_clients[0].test(metrics['test'])  #modo per valutare il training appena fatto (ma non è che dobbiamo fare train_clients[0].test???)
        # train_clients[0].test(metrics['test'])
        # metrics['test'].get_results()
        print(metrics['test'])


        print(metrics)

        # server = Server(args, train_clients, test_clients, model, metrics)
        # server.train()

    print("Centralized END")
    '''
    # DUBBIONE: a pag. 7 esperimenti sulla selezione dei client
    vengono poste delle probabilità di selezione 
    10% client -> prob. .5
    30% client -> prob. .0001

    -> queste sono da considerare probabilità aggregate di selezionare quel gruppo e poi all' interno di ognuno si sceglie uniformemente?
    -> questo porterebbe ad avere un gruppo di 60% -> prob. .4999 -> circa il 70% starebbe a prob. 1, però suddiviso 10%->p=.5 e 60%->p=.5 e poi uniformemente all'interno di ognuno


    -> o sono da considerare probabilità individuali di essere scelti con un tiro di moneta non equilibrata?
    -> se così fosse come questo va a combinarsi con il numero fisso di client da selezionare ad ogni round?? -> problema di definizione delle probabilità!!!
    --->  QUINDI NON QUESTO!
    '''


if __name__ == '__main__':
    main()