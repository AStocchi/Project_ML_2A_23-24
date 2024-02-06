import copy
import torch
import matplotlib.pyplot as plt

from scipy import ndimage
from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        new_bs = min(len(self.dataset), self.args.bs)
        self.train_loader = DataLoader(self.dataset, batch_size=new_bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=True)  # test-batch_size originale era 1, shuffle = True sarebbe più sensato per l' evaluation di base è settato a False
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()
        self.test_client_flag = test_client

    def __str__(self):
        return self.name

    def compute_loss(self, output, labels, cost_function, cuda_device):
        labels = labels.to(cuda_device)
        loss = cost_function(output, labels)
        return loss

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18':
            return self.model(images)
        if self.args.model == 'Convnn':
            return self.model(images)  # self.model(images) e self.model.forward(images) sono scritture analoghe, danno lo stesso risultato

        raise NotImplementedError

    def get_cost_function(self):
        cost_function = torch.nn.CrossEntropyLoss()
        return cost_function

    def get_optimizer(self, lr, wd, momentum):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
        return optimizer

    def transform_image(self, y, dim_y):
        z = torch.reshape(y, (dim_y, 1, 28, 28))
        return z

    def only_output(self, z):
        outputs = self._get_outputs(z)
        return outputs
        
    def output_and_loss(self, z, labels, cost_function, cuda_device):
        outputs = self._get_outputs(z)
        # Apply the loss
        loss = self.compute_loss(outputs, labels, cost_function, cuda_device)
        return outputs, loss

    def run_epoch(self, cur_epoch, optimizer, cost_function):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """

        samples = 0
        cumulative_loss = 0
        cumulative_accuracy = 0
        self.model.train()
        for cur_step, (images, labels) in enumerate(self.train_loader):
            cuda_device = torch.device("cuda:0")

            # if cur_step%1000==0:
            #    print("  -epoch: " + str(cur_epoch) + "  -iteration: " + str(cur_step))

            # Forward pass
            y = torch.stack(images, dim=1)  # y[1] #immagine vettorizzata
            value_bs = y.size()
            
            z = self.transform_image(y, value_bs[0])

            z = z.float()
            z = z.to(cuda_device)

            # Reset the optimizer (spostato, va fatto prima del resto [in realtà probabilmente non cambia nulla])
            optimizer.zero_grad()

            outputs, loss = self.output_and_loss(z, labels, cost_function, cuda_device)

            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()

            # Better print something, no?
            samples += z.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            labels = labels.to(cuda_device)
            cumulative_accuracy += predicted.eq(labels).sum().item()
        if samples == 0:
            breakpoint()

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        cost_function = self.get_cost_function()

        optimizer = self.get_optimizer(self.args.lr, self.args.wd, self.args.m)  # TO DO fare tuning dei parametri

        for epoch in range(self.args.num_epochs):
            _, _ = self.run_epoch(epoch, optimizer, cost_function)


    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        self.model.train(mode=False)

        with torch.no_grad(): # la condizione "with torch.no_grad():" Disabilita il calcolo dei gradienti durante la valutazione di ciò che viene in seguito a with
            for i, (images, labels) in enumerate(self.test_loader):
                if not self.test_client_flag: 
                    if i>=15:  # se il client è di train e lancia .test() -> sta facendo evaluation e viene interrotto dopo 15 sample
                        break
                cuda_device = torch.device("cuda:0")
                y = torch.stack(images, dim=1)
                value_bs = y.size()
                z = self.transform_image(y, value_bs[0])
                z = z.float()
                z = z.to(cuda_device)
                outputs = self.only_output(z)
                labels = labels.to(cuda_device)
                self.update_metric(metric, outputs, labels)

