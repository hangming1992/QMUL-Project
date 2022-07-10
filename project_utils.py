import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
from pylab import rcParams

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics
from torchmetrics.functional import accuracy

import my_utils as mu



def test():
    print("import success")

def test2():
    print("ok")

######## Preprocess ######## 

def s_to_str(row):
    return str(row.Semanticity)

def db_to_str(row):
    return str(row.SNRdB)

def c_to_str(row):
    return str(row.Correctness)

def new_c_level(row):
    level = row.Correctness // 10
    if level == 10:
        return str(int(level - 1))
    else:
        return str(int(level))

def preprocess_Y2(Y):
    
#     Y = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Textscore.csv'.format(subject_No))
    caseid = list(Y['CaseID'].values)
    trial_No = list(range(1, 1+len(caseid)))
    Y['trial_No'] = trial_No
    
    Y = Y.assign(
    Semanticity_str = Y.apply(s_to_str, axis=1),
    SNRdB_str = Y.apply(db_to_str, axis=1),
    Correctness_str = Y.apply(new_c_level, axis=1)
    )
    
    label_encoder_correctness = LabelEncoder()
    label_encoder_SNRdB = LabelEncoder()
    label_encoder_Semanticity = LabelEncoder()
    encoded_labels_correctness = label_encoder_correctness.fit_transform(Y.Correctness_str)
    encoded_labels_SNRdB = label_encoder_SNRdB.fit_transform(Y.SNRdB_str)
    encoded_labels_Semanticity = label_encoder_Semanticity.fit_transform(Y.Semanticity_str)
    
    Y["Correctness_label"] = encoded_labels_correctness
    Y["SNRdB_label"] = encoded_labels_SNRdB
    Y["Semanticity_label"] = encoded_labels_Semanticity
    
    return Y

def preprocess_Y(subject_No):
    
    Y = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Textscore.csv'.format(subject_No))
    caseid = list(Y['CaseID'].values)
    trial_No = list(range(1, 1+len(caseid)))
    Y['trial_No'] = trial_No
    
    Y = Y.assign(
    Semanticity_str = Y.apply(s_to_str, axis=1),
    SNRdB_str = Y.apply(db_to_str, axis=1),
    Correctness_str = Y.apply(new_c_level, axis=1)
    )
    
    label_encoder_correctness = LabelEncoder()
    label_encoder_SNRdB = LabelEncoder()
    label_encoder_Semanticity = LabelEncoder()
    encoded_labels_correctness = label_encoder_correctness.fit_transform(Y.Correctness_str)
    encoded_labels_SNRdB = label_encoder_SNRdB.fit_transform(Y.SNRdB_str)
    encoded_labels_Semanticity = label_encoder_Semanticity.fit_transform(Y.Semanticity_str)
    
    Y["Correctness_label"] = encoded_labels_correctness
    Y["SNRdB_label"] = encoded_labels_SNRdB
    Y["Semanticity_label"] = encoded_labels_Semanticity
    
    return Y

def preprocess_XY(subject_No):
    
    X_clean = pd.read_csv('eeg_clean/{}.csv'.format(subject_No))
    X = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Signals.csv'.format(subject_No))
    Y = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Textscore.csv'.format(subject_No))
    
    caseid = list(Y['CaseID'].values)
    unique = list(Y['CaseID'].unique())

    substitute = list(range(11, 11+len(caseid)))
    trial_No = list(range(1, 1+len(caseid)))

    Y['Sub'] = substitute
    Y['trial_No'] = trial_No
    sub_trial_map = dict(zip(substitute, trial_No))

    X.iloc[:,1:15] = X_clean
    X = X.drop(X.loc[X['CaseID'] == -1].index)
    X.index = range(X.shape[0])
    
    print("Assigning trial ID:")
    start_index = 0
    # start_time = time.time()
    for indexY, rowY in Y.iterrows():
        i = rowY['CaseID']
        print(rowY['trial_No'], end = ' ')

        X_update = X[start_index :].copy()


        for indexX, rowX in X_update.iterrows():

            j = rowX.at['CaseID']

    #         if (indexX % 10000) == 0:
    #             print('X:', indexX, j)

    #         if j not in unique:
    #             continue



#             if j == i :

            # if same then substitute

#             print('old caseID: ', X.at[indexX,'CaseID'])
            X.at[indexX,'CaseID'] = rowY['Sub']
#             X.at[indexX,'trial_No'] = int(rowY['trial_No'])
#             print('new caseID: ', X.at[indexX,'CaseID'])

            # within X?
            if (indexX + 1) in X.index: 

                # detect boundary (different LWR stage)
                if rowX['Label_T'] == 2 and X.loc[indexX+1, 'Label_T'] == 0:
#                     print('new trial: ', indexX)
                    start_index = indexX + 1
                    break

            # break at the end of X
            else:
                print('Done')
                break

            continue

    #         if new_trial_bool and j != i:
    #             print('backup break')
    #             break

    # end_time = time.time()
    # print(end_time - start_time )
    
#     print(X.CaseID.unique())
    
    def SubToTrial(row):
        sub = row.at['CaseID']
#         print(sub)
#         print(sub_trial_map[sub])
        return sub_trial_map[sub]


    X = X.assign(Trial_No = X.apply(SubToTrial, axis = 1))

    useless = list(X.columns)[15:21]
    useless.append('CaseID')
    X = X.drop(useless, axis = 1)
    X.index = range(X.shape[0])
    
    ##  Y
    Y = Y.assign(
    Semanticity_str = Y.apply(s_to_str, axis=1),
    SNRdB_str = Y.apply(db_to_str, axis=1),
    Correctness_str = Y.apply(c_to_str, axis=1)
    )
    
    label_encoder_correctness = LabelEncoder()
    label_encoder_SNRdB = LabelEncoder()
    label_encoder_Semanticity = LabelEncoder()
    encoded_labels_correctness = label_encoder_correctness.fit_transform(Y.Correctness_str)
    encoded_labels_SNRdB = label_encoder_SNRdB.fit_transform(Y.SNRdB_str)
    encoded_labels_Semanticity = label_encoder_Semanticity.fit_transform(Y.Semanticity_str)
    
    Y["Correctness_label"] = encoded_labels_correctness
    Y["SNRdB_label"] = encoded_labels_SNRdB
    Y["Semanticity_label"] = encoded_labels_Semanticity
    
    return X,Y



    
######## Dataset and DataModuel ######## 

def Split_Interpolate(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        feature = group.iloc[:,1:15].to_numpy()

        # Interpolate to the same size 
        feature = torch.tensor(feature.reshape(1,-1,14))
        feature = interpolate(torch.transpose(feature,1,2), max_trial_length)
        feature = feature.reshape(-1,14).numpy()
    #     pad = max_trial_length - group.shape[0]
    #     pad_spec = ((0,pad),(0,0))
    #     feature = np.pad(feature, pad_spec, 'constant', constant_values=0)

        # label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S']  #同一个trial No对应好几条数据,随便选，选第一个

        label = Y[Y.trial_No == t_No].iloc[0][target_label]
        # label = Y[Y.trial_No == t_No].iloc[0]["Semanticity_label"]
        # label = Y[Y.trial_No == t_No].iloc[0]["Correctness_label"]

    #     X_trial.append((feature, label))
        X_trial.append((feature, label)) # for inter-subject 堆tensor要同dimension
    
    return X_trial







def Aug_data(X_trial):
    backup_X = X_trial.copy()
    ori_length = len(X_trial)
    print('Size before Aug:', ori_length)
    
    for feature, label in backup_X:
        for scale in np.arange(0.9, 1.1, 0.01):
            aug_data = feature * scale
            for col in range(feature.shape[1]):
                noise = np.random.normal(feature[:,col].mean(), 1/scale, size = feature.shape[0])
                aug_data[:,col] += noise
            X_trial.append((aug_data, label))

    print('Done. Size after Aug:',len(X_trial))


class PhyAAtDataset(Dataset):
    
    def __init__(self, X_trial):
        self.X_trial = X_trial
        
    def __len__(self):
        return len(self.X_trial)
    
    # def shape(self):
    #     if self.X_trial[0][0].shape != self.X_trial[1][0].shape:
    #         assert 'different shape'
    #     return len(self.X_trial), self.X_trial[0][0].shape[0], self.X_trial[0][0].shape[1]
    
    def __getitem__(self, idx):
        feature,label = self.X_trial[idx]

        feature = torch.tensor(feature).float()
        label = torch.tensor(label).long()
        return dict(
            feature = feature,
            label = label
        )

    
class PhyAAtDataModule(pl.LightningDataModule):
    
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        
    def setup(self, stage=None):
        self.train_dataset = PhyAAtDataset(self.train_sequences)
        self.test_dataset = PhyAAtDataset(self.test_sequences)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True  
            ,num_workers = 4
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            ,num_workers = 4
        )
    
    def test_dataloader(self): 
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            shuffle = False
            ,num_workers = 4
        )
    

######## Net and Model ######## 

class RNNnet(torch.nn.Module):
    def __init__(self, num_feature, num_classes, num_hidden = 64, num_layers = 3):
        super(RNNnet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size = num_feature,
            hidden_size = num_hidden,
            num_layers = num_layers,
            batch_first = True
        )
        self.classifier = nn.Linear(num_hidden, num_classes)

    def forward(self, x):
        # self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
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