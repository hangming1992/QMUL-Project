import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.functional import accuracy



######## Net and Model ######## 

def flatten_size(max_trial_length):
    dummy_input = torch.rand(size=(1, 14, max_trial_length), dtype=torch.float32)
    net = nn.Sequential(
    nn.BatchNorm1d(14)
    ,nn.Conv1d(14, 84, kernel_size = 128*3, stride = 2, padding = 0)
    ,nn.AvgPool1d(kernel_size = 64, stride = 2)
    ,nn.Conv1d(84, 200, kernel_size = 32, stride = 4, padding = 0)
    ,nn.AvgPool1d(kernel_size = 4, stride = 8)
    ,nn.LeakyReLU()
    ,nn.Dropout()
    ,nn.Flatten()
)
    hid_size = net(dummy_input).shape[1]
    
    return hid_size



class CNNnet(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, flatten_size):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Conv1d(num_inputs, 84, kernel_size = 128*3, stride = 2, padding = 0)
        self.conv2 = nn.Conv1d(84, 200, kernel_size = 32, stride = 4, padding = 0)

        self.rl = nn.LeakyReLU()
        
        self.avgpool1 = nn.AvgPool1d(kernel_size = 64, stride = 2)
        self.avgpool2 = nn.AvgPool1d(kernel_size = 4, stride = 8)
        self.drop = nn.Dropout()
        
        self.bn0 = nn.BatchNorm1d(num_inputs)
        # self.bn1 = nn.BatchNorm1d(num_inputs)
        self.bn1 = nn.BatchNorm1d(84)
        # self.bn2 = nn.BatchNorm1d(84)
        self.bn2 = nn.BatchNorm1d(200)

        self.fl = nn.Flatten()
        self.linear1 = nn.Linear(flatten_size,100)
        self.linear2 = nn.Linear(100, num_outputs)

    def forward(self, x):
        out = x

        out = self.bn0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.rl(out)
        out = self.avgpool1(out)
        out = self.drop(out)
        
        # out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.rl(out)
        out = self.avgpool2(out)

        out = self.fl(out)
        out = self.linear1(out)
        out = self.linear2(out)

        return out
    

    

class CNNmodel(pl.LightningModule):
    def __init__(self, num_features: int, num_classes: int, flatten_size:int):
        super().__init__()
        self.model = CNNnet(num_features, num_classes, flatten_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels = None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output
    
    def training_step(self, batch, batch_idx):
        sequences = batch['feature']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('train_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss':loss, 'accuracy': step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences = batch['feature']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss':loss, 'accuracy': step_accuracy}
        
    def test_step(self, batch, batch_idx):
        sequences = batch['feature']
        labels = batch['label']
        loss, outputs = self(sequences, labels)
        predictions = torch.argmax(outputs, dim = 1)
        step_accuracy = accuracy(predictions, labels)
        
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_accuracy', step_accuracy, prog_bar=True, logger=True)
        return {'loss':loss, 'accuracy': step_accuracy}
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr = 0.001 , weight_decay =0)