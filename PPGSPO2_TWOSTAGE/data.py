import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader  
from torch.optim import lr_scheduler
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import lightgbm as lgb

def set_device(config):
    if config['use_cuda']:
        device = torch.device('cuda')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['device_num'])
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    print("==========================================")
    print("Use Device :", device)
    print("Available cuda devices :", torch.cuda.device_count())
    print("Current cuda device :", torch.cuda.current_device())
    print("Name of cuda device :", torch.cuda.get_device_name(device))
    print("==========================================")
    


def ppg_dataloader(config):
    
    '''
    1. return:
    x_dataloaders[0]: train 관련 ([0][0]: binary, [0][1]: sec1, [0][2]: sec2 // x_dataloaders[1]: test 관련)
    2. 22.08.03: 아직 환경별 testset 미구성, multi-set 확장 미고려
    '''
    # CSV파일에서 dataframe으로 읽어오기
    train_data_ir, train_data_red, test_data_ir, test_data_red, train_data_spo2, test_data_spo2 \
        = dataframe_from_csv(config)
        
    print("reading csv complete.")
    
    # ir, red 신호 인덱스
    ppg_ir_index = ["ssa_ir_{0:03d}".format(file_ind) for file_ind in range(1,706)]
    ppg_red_index = ["ssa_red_{0:03d}".format(file_ind) for file_ind in range(1,706)]
    # ppg_ir_index = ["ppg_ir_{0:03d}".format(file_ind) for file_ind in range(1,751)]
    # ppg_red_index = ["ppg_red_{0:03d}".format(file_ind) for file_ind in range(1,751)]

    x_train_ppg_ir = copy.deepcopy(train_data_ir[ppg_ir_index].values)
    x_test_ppg_ir = copy.deepcopy(test_data_ir[ppg_ir_index].values)

    x_train_ppg_red = copy.deepcopy(train_data_red[ppg_red_index].values)
    x_test_ppg_red = copy.deepcopy(test_data_red[ppg_red_index].values)

    y_train_spo2 = copy.deepcopy(train_data_spo2[["spo2_ref"]].values)
    y_test_spo2 = copy.deepcopy(test_data_spo2[["spo2_ref"]].values)

    y_train_binary = copy.deepcopy(train_data_spo2[["binary"]].values)
    y_test_binary = copy.deepcopy(test_data_spo2[["binary"]].values)
    '''
    x_train_ppg_ir_sec1 = x_train_ppg_ir[train_data_spo2["binary"] ==1]
    x_train_ppg_ir_sec2 = x_train_ppg_ir[train_data_spo2["binary"] ==0]
    x_train_ppg_red_sec1 = x_train_ppg_red[train_data_spo2["binary"] ==1]
    x_train_ppg_red_sec2 = x_train_ppg_red[train_data_spo2["binary"] ==0]

    x_test_ppg_ir_sec1 = x_test_ppg_ir[test_data_spo2["binary"] ==1]
    x_test_ppg_ir_sec2 = x_test_ppg_ir[test_data_spo2["binary"] ==0]
    x_test_ppg_red_sec1 = x_test_ppg_red[test_data_spo2["binary"] ==1]
    x_test_ppg_red_sec2 = x_test_ppg_red[test_data_spo2["binary"] ==0]

    y_train_spo2_sec1 = x_test_ppg_red[test_data_spo2["binary"] ==1]
    y_train_spo2_sec2 = x_test_ppg_red[test_data_spo2["binary"] ==0]
    '''
    print("processing ppg complete.")

    # Dataloader 구축 (LGBM / 1dCNN)
    x_train = np.concatenate([x_train_ppg_ir, x_train_ppg_red], 1)
    x_test = np.concatenate([x_test_ppg_ir, x_test_ppg_red], 1)

    x_train_sec1 = x_train[train_data_spo2["binary"] ==1]
    y_train_sec1 = y_train_spo2[train_data_spo2["binary"] ==1]
    x_train_sec2 = x_train[train_data_spo2["binary"] ==0]
    y_train_sec2 = y_train_spo2[train_data_spo2["binary"] ==0]

    x_test_sec1 = x_test[test_data_spo2["binary"] ==1]
    y_test_sec1 = y_test_spo2[test_data_spo2["binary"] ==1]
    x_test_sec2 = x_test[test_data_spo2["binary"] ==0]
    y_test_sec2 = y_test_spo2[test_data_spo2["binary"] ==0]
    
    # for LGBM
    train_binary_lgbm = lgb.Dataset(x_train, label = y_train_binary) 
    train_sec1_lgbm = lgb.Dataset(x_train_sec1, label = y_train_sec1)
    train_sec2_lgbm = lgb.Dataset(x_train_sec2, label = y_train_sec2)

    test_binary_lgbm = lgb.Dataset(x_test, label = y_test_binary) 

    lgbm_dataloaders = [[train_binary_lgbm, train_sec1_lgbm, train_sec2_lgbm], test_binary_lgbm]

    # for CNN
    x_train_cnn = torch.FloatTensor(x_train)
    y_train_spo2_cnn = torch.FloatTensor(y_train_spo2)
    y_train_binary_cnn = torch.FloatTensor(y_train_binary)

    x_test_cnn = torch.FloatTensor(x_test)
    y_test_spo2_cnn = torch.FloatTensor(y_test_spo2)
    y_test_binary_cnn = torch.FloatTensor(y_test_binary)

    x_train_sec1_cnn = torch.FloatTensor(x_train_sec1)
    y_train_sec1_cnn = torch.FloatTensor(y_train_sec1)
    x_train_sec2_cnn = torch.FloatTensor(x_train_sec2)
    y_train_sec2_cnn = torch.FloatTensor(y_train_sec2)

    x_test_sec1_cnn = torch.FloatTensor(x_test_sec1)
    y_test_sec1_cnn = torch.FloatTensor(y_test_sec1)
    x_test_sec2_cnn = torch.FloatTensor(x_test_sec2)
    y_test_sec2_cnn = torch.FloatTensor(y_test_sec2)
    
    batch_size = config['batch_size']  # for both train and testidation batch size
    test_batch_size = config['test_batch_size']

    dataset_binary_cnn = TensorDataset(x_train_cnn, y_train_binary_cnn)
    dataset_sec1_cnn = TensorDataset(x_train_sec1_cnn, y_train_sec1_cnn)
    dataset_sec2_cnn = TensorDataset(x_train_sec2_cnn, y_train_sec2_cnn)

    train_dataloader_binary = DataLoader(dataset_binary_cnn, batch_size=batch_size, shuffle=True)
    train_dataloader_sec1 = DataLoader(dataset_sec1_cnn, batch_size=batch_size, shuffle=True)
    train_dataloader_sec2 = DataLoader(dataset_sec2_cnn, batch_size=batch_size, shuffle=True)
    
    # test dataloader..어떻게 하지?
    test_spo2_cnn = TensorDataset(x_test_cnn, y_test_spo2_cnn)
    test_spo2_cnn = DataLoader(test_spo2_cnn, batch_size=test_batch_size, shuffle=True)
    test_binary_cnn = TensorDataset(x_test_cnn, y_test_binary_cnn)
    test_binary_cnn = DataLoader(test_binary_cnn, batch_size=test_batch_size, shuffle=True)

    cnn_dataloaders = [[train_dataloader_binary,train_dataloader_sec1,train_dataloader_sec2], [test_binary_cnn, test_spo2_cnn]]
    
    print("DATALOAD COMPLETE")

    return lgbm_dataloaders, cnn_dataloaders


def dataframe_from_csv(config):
    train_data_ir = pd.read_csv(config['train_data_ir'])
    test_data_ir = pd.read_csv(config['test_data_ir'])

    train_data_red = pd.read_csv(config['train_data_red'])
    test_data_red = pd.read_csv(config['test_data_red'])

    train_data_spo2 = pd.read_csv(config['train_data_spo2'])
    test_data_spo2 = pd.read_csv(config['test_data_spo2'])     

    # binary label 생성
    train_data_spo2.loc[train_data_spo2['spo2_ref'] >= config['criteria'], 'binary'] = 1
    train_data_spo2.loc[train_data_spo2['spo2_ref'] < config['criteria'], 'binary'] = 0   
    test_data_spo2.loc[test_data_spo2['spo2_ref'] >= config['criteria'], 'binary'] = 1
    test_data_spo2.loc[test_data_spo2['spo2_ref'] < config['criteria'], 'binary'] = 0  

    return train_data_ir, train_data_red, test_data_ir, test_data_red, train_data_spo2, test_data_spo2





