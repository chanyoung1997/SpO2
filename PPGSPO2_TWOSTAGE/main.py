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
import argparse
from data import *
from model import *
from eval import *


# Read settings
with open('config.yml', mode='r', encoding='utf8') as file:
    config = yaml.load(file, Loader=yaml.Loader)

#LIGHTGBM_CONFIG
params_bin = {'learning_rate': 0.01, 
          'max_depth': -1, 
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'AUC', 
          'is_training_metric': True, 
          'num_leaves': 30, 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.7, 
          'bagging_freq': 5, 
          'seed':2018}

params = {'learning_rate': 0.01, 
          'max_depth': -1, 
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'AUC', 
          'is_training_metric': True, 
          'num_leaves': 30, 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.7, 
          'bagging_freq': 5, 
          'seed':2018}


if __name__ == "__main__":  
    '''
    lgbm_dataloaders[0]: training  [0]: bin [1]: sec1 [2]: sec2
    lgbm_dataloaders[1]: test
    '''

    lgbm_dataloaders, cnn_dataloaders = ppg_dataloader(config)
    
    # plug_play
    model0 = config['model0']
    model1 = config['model1']
    model2 = config['model2']
    
    # Training
    ## 이진 분류기
    if model0 == 'lgbm':
        model_bin = lgb.train(params_bin, lgbm_dataloaders[0][0], 10, lgbm_dataloaders[1], verbose_eval=100, early_stopping_rounds=100)
        print("----------------------BINARY CLF TRAIN COMPLETE-------------------")
    elif model0 == 'cnn':
        pass                
    # Sec 1
    # Sec1, Sec2 validation set 미구축
    if model1 == 'lgbm':
        model_reg1 = lgb.train(params, lgbm_dataloaders[0][1], 10, verbose_eval=100)
        print("----------------------SEC1 CLF TRAIN COMPLETE-------------------")
    elif model1 == 'cnn':
        pass      
    # Sec 2
    if model2 == 'lgbm':
        model_reg2 = lgb.train(params, lgbm_dataloaders[0][2], 10, verbose_eval=100)
        print("----------------------SEC2 CLF TRAIN COMPLETE-------------------")
    elif model2 == 'cnn':
        pass  
    # Test
    
    
    # Evaluate





