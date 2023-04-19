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

class RNNnet(torch.nn.Module):
    def __init__(self, num_feature, num_classes, num_hidden = 84, num_layers = 3):
        super(RNNnet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size = num_feature,
            hidden_size = num_hidden,
            num_layers = num_layers,
            batch_first = True
            dropout = 0.5
        )
        self.classifier = nn.Linear(num_hidden, num_classes)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)


    def forward(self, x):
        # self.lstm.flatten_parameters()
        
        lengths = [x_.size(0) for x_ in x]
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True, enforce_sorted=True)
        
        _, (hidden, _) = self.lstm(x_packed)
        out = hidden[-1]
        out = self.classifier(out)
        return out
    

    

class RNNmodel(pl.LightningModule):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.model = RNNnet(num_features, num_classes)
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
    

    
