import argparse
import torch
import os
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
from sklearn.metrics import confusion_matrix
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


def evaluate_consensus( dataset, model_list, output, study ):
    """Obtain the consensus predictions by running them through all
    5 test models that we have trained on. Note, each model is trained 
    on a different training set, so even though the weights are different 
    each model would necessarily have the same number of hyperparameters
    ** Parameters **
    dataloader (Dataloader object) : The dataset input via a Dataloader
    model_list (list,Module object) : The list of Models for consensus modeling

    """

    n_models = len(model_list)
    test_dataset = PerResidueDataset(dataset)
    dataloader = data.DataLoader( test_dataset, batch_size=32, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )

    data_list = []

    models = []
    model_preds = []
    tokens = []

    for i in range(n_models):
        models.append(torch.load(model_list[i], map_location='cpu'))
        models[-1].eval()
        model_preds.append([])

    with torch.no_grad():
        for seq_inputs in tqdm(dataloader):
            inputs = seq_inputs[0]
            sequence_input = inputs[0]
            enr_input = inputs[1]
            token = seq_inputs[2]
            for i in range(len(models)):
                model_preds[i].append( F.softmax( models[i](sequence_input, enr_input), -1 ) )
            tokens.append(token)
    
    tokens = list(chain(*tokens))
    for i in range(len(model_preds)):
        model_preds[i] = torch.cat(model_preds[i])
    

    for i in range(len(tokens)):
        token_id = str(tokens[i]).replace("'","").split('_')
        if( len(token_id) == 3 ):
            res_id = token_id[-2][:-1]
            res_chain = token_id[-2][-1]
            res_aa = token_id[-1]
            name = 'complex_'+res_id+res_chain+'_'+res_aa
            bin1 = np.average([model_preds[0][i][0], model_preds[1][i][0], 
                            model_preds[2][i][0] ])
            bin2 = np.average([model_preds[0][i][1], model_preds[1][i][1], 
                            model_preds[2][i][1] ])
            bin3 = np.average([model_preds[0][i][2], model_preds[1][i][2], 
                            model_preds[2][i][2] ])
            bin4 = np.average([model_preds[0][i][3], model_preds[1][i][3], 
                            model_preds[2][i][3] ])
            
            label = np.argmax([bin1, bin2, bin3, bin4])
            
            score_terms = [ study, name, bin1, bin2, bin3, bin4, label ]
            data_list.append(score_terms)
        else: continue

    data_list = list(map( list, zip(*data_list) ))    
    df = pd.DataFrame(np.transpose(data_list))
    df.columns = [ 'Study', 'Name', 'under50-bin', '50-60-bin', '60-70-bin', '70up-bin', 'Predictions' ]
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
    parser.add_argument('--dataset', type=str, 
                        help='Name of the mutant dataset')
    parser.add_argument('--study', type=str, default='test',
                        help='Name of the experiment/complex study')
    
    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    output = args.output
    dataset = args.dataset
    study = args.study
    model_path = args.model_path

    model_list = [ model_path + "ensemble/Model1.torch",
                   model_path + "ensemble/Model2.torch",
                   model_path + "ensemble/Model3.torch" ]

    evaluate_consensus(dataset, model_list, output, study)

if __name__ == '__main__':
    _cli()








    


