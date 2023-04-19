import pandas as pd
import numpy as np
# from tqdm.auto import tqdm

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



######## Dataset and DataModuel ######## 




class PhyAAtDataset(Dataset):
    
    def __init__(self, X_trial):
        self.X_trial = X_trial
        
    def __len__(self):
        return len(self.X_trial)
    
    def __getitem__(self, idx):
        feature,label = self.X_trial[idx]
        feature = feature.float()
        label = label.long()
        return dict(
            feature = feature,
            label = label
        )

    
class PhyAAtDataModule(pl.LightningDataModule):
    
    def __init__(self, train_sequences, valid_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.valid_sequences = valid_sequences
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = PhyAAtDataset(self.train_sequences)
        self.valid_dataset = PhyAAtDataset(self.valid_sequences)
        self.test_dataset = PhyAAtDataset(self.test_sequences)
        
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True  
            ,num_workers = 4
            ,persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size = self.batch_size,
            shuffle = False
            ,num_workers = 4
            ,persistent_workers=True
        )
    
    def test_dataloader(self): 
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            ,num_workers = 4
            ,persistent_workers=True
        )

    



# # Data Augmentation. Optional

# def Aug_data(X_trial):
#     backup_X = X_trial.copy()
#     ori_length = len(X_trial)
#     print('Size before Aug:', ori_length)
    
#     for feature, label in backup_X:
#         for scale in np.arange(0.9, 1.1, 0.01):
#             aug_data = feature * scale
#             for col in range(feature.shape[1]):
#                 noise = np.random.normal(feature[:,col].mean(), 1/scale, size = feature.shape[0])
#                 aug_data[:,col] += noise
#             X_trial.append((aug_data, label))

#     print('Done. Size after Aug:',len(X_trial))
    
