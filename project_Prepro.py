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



######## Preprocess ######## 

# for X: Assign new Trial ID
# for Y: assign task label
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
    
    # Assign New Trial ID
    print("Assigning trial ID:")
    start_index = 0
    for indexY, rowY in Y.iterrows():
        i = rowY['CaseID']
        print(rowY['trial_No'], end = ' ')
        X_update = X[start_index :].copy()
        for indexX, rowX in X_update.iterrows():
            j = rowX.at['CaseID']
            X.at[indexX,'CaseID'] = rowY['Sub']
            if (indexX + 1) in X.index: 
                if rowX['Label_T'] == 2 and X.loc[indexX+1, 'Label_T'] == 0:
                    start_index = indexX + 1
                    break
            # break at the end of X
            else:
                print('Done')
                break

            continue

            
    def SubToTrial(row):
        sub = row.at['CaseID']
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

def preprocess_Y(subject_No):
    
    Y = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Textscore.csv'.format(subject_No))
    caseid = list(Y['CaseID'].values)
    trial_No = list(range(1, 1+len(caseid)))
    Y['trial_No'] = trial_No
    
    # Assign label
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

def preprocess_Y2(Y):
    """
    input Y instead of subject No. For all subject
    """
#     Y = pd.read_csv('PhyAAt_Data/phyaat_dataset/Signals/S{0}/S{0}_Textscore.csv'.format(subject_No))
    caseid = list(Y['CaseID'].values)
    trial_No = list(range(1, 1+len(caseid)))
    Y['trial_No'] = trial_No
    
    # Assign label
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

def preprocess_Y_TFM(subject_No):
    
    Y = pd.read_csv('S{0}_Textscore.csv'.format(subject_No))
    caseid = list(Y['CaseID'].values)
    trial_No = list(range(1, 1+len(caseid)))
    Y['trial_No'] = trial_No
    
    # Assign label
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



########### Split and Pad functions ##########

    
def Split_Pad_CNN(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        own_length = group.shape[0]
        pad_length = max_trial_length - own_length
        feature = group.iloc[:,1:15].to_numpy()
        feature = torch.tensor(feature).transpose(0,1)
        # pad on first dimension only on the right
        feature = F.pad(feature, (0,pad_length))
        # shape: (14, length)
        label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S'] 
        label = torch.tensor(label)
        X_trial.append((feature, label)) 
    
    return X_trial, max_trial_length    
    

def Split_Interpolate_CNN(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        feature = group.iloc[:,1:15].to_numpy()
        feature = torch.transpose(torch.tensor(feature.reshape(1,-1,14)),1,2)
        # expected inputs are 3-D, 4-D or 5-D in shape. so it will interpolate the last dimension when input is 3D
        feature = interpolate(feature, max_trial_length)
        feature = feature.reshape(14,-1)
        
        label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S']  
        label = torch.tensor(label)
        X_trial.append((feature, label))
    
    return X_trial, max_trial_length


def Split_Interpolate_RNN(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        feature = group.iloc[:,1:15].to_numpy()

        # Interpolate to the same size 
        feature = torch.tensor(feature.reshape(1,-1,14))
        feature = interpolate(torch.transpose(feature,1,2), max_trial_length)
        feature = feature.reshape(-1,14)

        label = Y[Y.trial_No == t_No].iloc[0][target_label]
        label = torch.tensor(label)

        X_trial.append((feature, label)) 
    
    return X_trial


def Split_Pad(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        own_length = group.shape[0]
        pad_length = max_trial_length - own_length
        feature = group.iloc[:,1:15].to_numpy()

        feature = torch.tensor(feature).transpose(0,1)
        # pad on first dimension only on the right
        feature = F.pad(feature, (0,pad_length))
        feature = feature.transpose(0,1)
        label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S'] 
        label = torch.tensor(label)
        
        X_trial.append((feature, label))
    
    return X_trial


def Split(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    X_trial = []
    for t_No, group in trial_group:

        feature = group.iloc[:,1:15].to_numpy()
        feature = torch.tensor(feature)
        label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S']  
        X_trial.append((feature, label)) 
    
    return X_trial




def Split_Pad_TFM(X, Y, target_label):
    trial_group = X.groupby('Trial_No')
    trial_length = [group[1].shape[0] for group in trial_group]
    max_trial_length = max(trial_length)
    
    X_trial = []
    for t_No, group in trial_group:

        own_length = group.shape[0]
        pad_length = max_trial_length - own_length
        feature = group.iloc[:,1:15].to_numpy()

        # input dimension :  (batch, seq, feature) with Batch_First = True
        feature = torch.tensor(feature).transpose(0,1)
        # pad on first dimension only on the right
        feature = F.pad(feature, (0,pad_length))
        feature = feature.transpose(0,1)
        
        label = group.loc[group['Trial_No'] == t_No].iloc[0]['Label_S']
        label = torch.tensor(label)
        X_trial.append((feature, label)) 
    
    return X_trial, max_trial_length

