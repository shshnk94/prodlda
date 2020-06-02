import argparse
import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

parser = argparse.ArgumentParser(description = 'prodlda cross validation wrapper')

parser.add_argument('--dataset', type=str, default='20ng', help='name of corpus')
parser.add_argument('--topics', type=str, default='50', help='number of topics')
parser.add_argument('--data_path', type=str, help='path to a fold of data')
parser.add_argument('--save_path', type=str, help='save path for every run')
parser.add_argument('--gpu', type=str, help='index of the gpu core which would contain this model')

args = parser.parse_args()

if args.dataset == '20ng':
    data_path = 'autoencoding_vi_for_topic_models\/data\/20news_clean'
else:
    data_path = 'autoencoding_vi_for_topic_models\/data\/nips'


def run_script(params, fold):

    os.system('CUDA_VISIBLE_DEVICES={} python prodlda/run.py -m prodlda'.format(args.gpu) +
              ' -f 100' + 
              ' -s 100' +
              ' -t ' + args.topics +
              ' -b 4096' + 
              ' -r ' + params['lr'] +
              ' -e ' + params['epochs'] +
              ' --fold ' + str(fold) +
              ' --mode train' +
              ' --data_path ' + args.data_path + 
              ' --save_path ' + args.save_path + ' &')

#Hyperparameters
hyperparameters = {'epochs': ['5000'],
                   'lr': ['5e-3', '5e-4']}#, '5e-5']}

for params in ParameterGrid(hyperparameters):
    for fold in range(3): #Hard coded values of fold
        run_script(params, fold)
