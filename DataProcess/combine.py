# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:48:46 2020

@author: Anxbbq
"""

import pandas as pd
import os
import csv
import numpy as np
import argparse

def get_args():
    ''' This function parses and return arguments passed in'''
    parser = argparse.ArgumentParser(description='Parameters for process data')
    parser.add_argument('-d', '--data_set', type=str, help='Dataset: Home_and_Kitchen, Movies_and_TV, Electronics, CDs_and_Vinyl', required=True)
    args = parser.parse_args()
    return args
args=get_args()
dataSet=args.data_set
TPS_DIR='../data/%s' % dataSet
inputfile_csv_1=TPS_DIR+'/%s_test.csv' % dataSet
inputfile_csv_2=TPS_DIR+'/%s_train.csv' % dataSet
inputfile_csv_3=TPS_DIR+'/%s_valid.csv' % dataSet
outputfile=TPS_DIR+'/%s_all.csv' % dataSet
f1 = open(inputfile_csv_1,'rb').read()
f2 = open(inputfile_csv_2,'rb').read()
with open(outputfile,'ab') as f: #将结果保存为result.csv
    f.write(f1)
    f.write(f2)