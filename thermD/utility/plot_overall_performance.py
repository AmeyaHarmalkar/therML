import argparse
import torch
import os, pathlib
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from itertools import chain
import h5py
import tqdm
from tqdm import tqdm
from datetime import datetime


# Model evaluation
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


import thermD
from thermD.datasets import PerResDataset
from thermD.datasets.PerResDataset import PerResidueDataset
from thermD.networks.MiniCNNs import *
from thermD.preprocessing.generate_h5_file import sequences_to_h5
from thermD.utility.get_spearmann_corr import evaluate_consensus, evaluate_consensus_mae


def get_filelist( direc ):
    """This funciton is just to obtain all the files in a directory that would be evaluated
    """
    file_list = []
    directory = os.path.join( direc )
    print( directory )
    for h5_file in os.listdir( directory ):
        if ( pathlib.Path(h5_file).suffix == '.h5' ):
            if os.path.exists( os.path.join(directory, h5_file) ):
                file_list.append( os.path.join(directory, h5_file) )
    
    return file_list



def get_all_spearmann( direc, model_list, output, method='Supervised', err_coeff = 'rho', out_preds = False ):

    file_list = get_filelist( direc )
    
    data_list = []

    for h5_file in file_list:
        label = h5_file.split()[0]
        rho_net = []
        rho_base = []
        for i in range(5):
            if ( err_coeff == 'mae' ):
                mae_val, base = evaluate_consensus_mae( h5_file, model_list, output, label, out_preds = False )
                rho_net.append( mae_val )
                rho_base.append( base )
            else:
                rho_net.append( evaluate_consensus( h5_file, model_list, output, label, out_preds = False )  )
                rho_base.append( 0.0 )
        rho = np.mean( rho_net )
        rho_std = np.std( rho_net )
        rho_b = np.mean( rho_base )
        data_list.append( [method, label, label, rho, rho_std, rho_b ] )

    data_list = list(map( list, zip(*data_list) ))    
    df = pd.DataFrame(np.transpose(data_list))
    print(df)
    df.columns = [ 'Method', 'Group', 'Label', 'rho', 'rho_stdev', 'rho_baseline' ]
    df.to_csv( output +'.csv', index=False )



def _get_args():
    """Obtain the command line arguments"""
    desc = ('''
        Script for predicting the bins for point mutants with 
        consensus across models trained on different datasets
    ''')

    parser = argparse.ArgumentParser(description=desc)
    
    #training arguments
    parser.add_argument('--output', type=str, default='output',
                        help='name of the output file')
    parser.add_argument('--model_path', type=str,
                        help='Path of the models')
    parser.add_argument('--direc', type=str, 
                        help='Name of the directory with required h5py file')
    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--err_coeff',  type=str, default='rho',
                        help='Type of error coefficient we need, MAE or Spearmanns rho. By default we will output rho')

    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    output = args.output
    direc = args.direc
    model_path = args.model_path
    err_coeff = args.err_coeff

    model_list = [ model_path + "ensemble/Model1.torch",
                   model_path + "ensemble/Model2.torch",
                   model_path + "ensemble/Model3.torch" ]

    get_all_spearmann( direc, model_list, output, err_coeff = err_coeff )

if __name__ == '__main__':
    _cli()


