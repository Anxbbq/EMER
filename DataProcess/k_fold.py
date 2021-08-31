# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:15:46 2020

@author: Anxbbq
"""


import numpy as np
import csv
from sklearn.model_selection import KFold
import pandas as pd
import os
import argparse

def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Parameters for process data')
    parser.add_argument('-d', '--data_set', type=str, help='Dataset: Home_and_Kitchen, Movies_and_TV, Electronics, CDs_and_Vinyl', required=True)
    args = parser.parse_args()
    return args

args = get_args()
dataSet=args.data_set
TPS_DIR='../data/%s' % dataSet
data_all = pd.read_csv(TPS_DIR+'/%s_all.csv' % dataSet,header=None)
print(data_all.shape)
data=data_all.values
#data=np.array(data_all.values.tolist(), dtype = str)
kf=KFold(n_splits=9)
i=0
for train_index, test_index in kf.split(data):
    if not os.path.exists(TPS_DIR + '/split_data_' + str(i) + '/'):
            os.makedirs(TPS_DIR + '/split_data_' + str(i) + '/')
    with open(TPS_DIR +'/split_data_'+str(i)+'/train.csv','w',newline="",encoding="utf-8") as f1:
        writer = csv.writer(f1)
        writer.writerows(data[train_index])
    with open(TPS_DIR +'/split_data_'+str(i)+'/test.csv','w',newline="",encoding="utf-8") as f2:
        writer = csv.writer(f2)
        writer.writerows(data[test_index])
    i=i+1
