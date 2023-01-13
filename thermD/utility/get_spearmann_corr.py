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
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, mean_absolute_error
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


def evaluate_consensus( dataset, model_list, output, study, out_preds = False ):
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

    def predict_ts50(value):
        """The mean_ts50 values are determined from the average TS50 value 
        of sequences in the TS50 dataset in each of the four bins i.e 
        [under 50, 50-60, 60-70, 70up]. We compute probabilities in each
        bin with our model and then equate the spearmann correlation coeff
        between this expected value and the actual value."""
        mean_ts50 = torch.tensor([36.0395, 56.6874, 64.190, 70.0740])
        expected_value = torch.dot(value, mean_ts50)
        return expected_value.numpy()


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
                value = F.softmax( models[i](sequence_input, enr_input), -1 )
                model_preds[i].append( value.cpu() )
            tokens.append(token)
    
    tokens = list(chain(*tokens))
    for i in range(len(model_preds)):
        model_preds[i] = torch.cat(model_preds[i])
    
    model_predictions = []

    for i in range(len(tokens)):
        
        bin1 = np.average([model_preds[0][i][0], model_preds[1][i][0], 
                        model_preds[2][i][0] ])
        bin2 = np.average([model_preds[0][i][1], model_preds[1][i][1], 
                        model_preds[2][i][1] ])
        bin3 = np.average([model_preds[0][i][2], model_preds[1][i][2], 
                        model_preds[2][i][2] ])
        bin4 = np.average([model_preds[0][i][3], model_preds[1][i][3], 
                        model_preds[2][i][3] ])
            
        model_predictions.append( torch.tensor( [ bin1, bin2, bin3, bin4 ] ) )

        label = np.argmax([bin1, bin2, bin3, bin4])
            
        score_terms = [ study, bin1, bin2, bin3, bin4, label ]
        data_list.append(score_terms)
   

    ts50_preds = []
    ts50_list = []
    for i in range(len(tokens)):
        exp = predict_ts50(model_predictions[i])
        ts50_preds.append(exp)
        ts50_list.append( [ tokens[i], exp ] )

    
    ts50_list = list(map(list,zip(*ts50_list)))
    tdf = pd.DataFrame(np.transpose(ts50_list), columns = [ 'Actual', 'Prediction' ] )
    rho = tdf[["Actual", "Prediction"]].corr(method="spearman").iloc[0::2,1::2].iloc[0,0].item()
    print("Spearmann corr : ", rho)

    if (out_preds):
        data_list = list(map( list, zip(*data_list) ))    
        df = pd.DataFrame(np.transpose(data_list))
        df.columns = [ 'Study', 'under50-bin', '50-60-bin', '60-70-bin', '70up-bin', 'Predictions' ]
        df.to_csv( output +'.csv', index=False )
    
    return rho


def evaluate_consensus_mae( dataset, model_list, output, study, out_preds = False ):
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

    def predict_ts50(value):
        """The mean_ts50 values are determined from the average TS50 value 
        of sequences in the TS50 dataset in each of the four bins i.e 
        [under 50, 50-60, 60-70, 70up]. We compute probabilities in each
        bin with our model and then equate the spearmann correlation coeff
        between this expected value and the actual value."""
        mean_ts50 = torch.tensor([36.0395, 56.6874, 64.190, 70.0740])
        argmax_vector = np.identity( len(np.array(value)) )[ :, np.array(value).argmax(0) ]
        expected_value = torch.dot(value, mean_ts50)
        mean_value = torch.dot( torch.FloatTensor(argmax_vector), mean_ts50 )
        return expected_value.numpy(), mean_value.numpy()


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
                value = F.softmax( models[i](sequence_input, enr_input), -1 )
                model_preds[i].append( value.cpu() )
            tokens.append(token)
    
    tokens = list(chain(*tokens))
    for i in range(len(model_preds)):
        model_preds[i] = torch.cat(model_preds[i])
    
    model_predictions = []

    for i in range(len(tokens)):
        
        bin1 = np.average([model_preds[0][i][0], model_preds[1][i][0], 
                        model_preds[2][i][0] ])
        bin2 = np.average([model_preds[0][i][1], model_preds[1][i][1], 
                        model_preds[2][i][1] ])
        bin3 = np.average([model_preds[0][i][2], model_preds[1][i][2], 
                        model_preds[2][i][2] ])
        bin4 = np.average([model_preds[0][i][3], model_preds[1][i][3], 
                        model_preds[2][i][3] ])
            
        model_predictions.append( torch.tensor( [ bin1, bin2, bin3, bin4 ] ) )

        label = np.argmax([bin1, bin2, bin3, bin4])
            
        score_terms = [ study, bin1, bin2, bin3, bin4, label ]
        data_list.append(score_terms)
   
    ts50_means = []
    ts50_preds = []
    ts50_list = []
    for i in range(len(tokens)):
        exp, mean_val = predict_ts50(model_predictions[i])
        ts50_preds.append(exp)
        ts50_means.append(mean_val)
        ts50_list.append( [ tokens[i], exp, mean_val ] )

    
    ts50_list = list(map(list,zip(*ts50_list)))
    tdf = pd.DataFrame(np.transpose(ts50_list), columns = [ 'Actual', 'Prediction', 'Mean' ] )
    actual_vals = tdf["Actual"].to_numpy()
    predicted_vals = tdf["Prediction"].to_numpy()
    mean_vals = tdf["Mean"].to_numpy()
    mae = mean_absolute_error( actual_vals, predicted_vals )
    baseline_mae = mean_absolute_error( actual_vals, mean_vals )
    print("Mean absolute error : ", mae)
    print("Baseline MAE : ", baseline_mae)


    if (out_preds):
        data_list = list(map( list, zip(*data_list) ))    
        df = pd.DataFrame(np.transpose(data_list))
        df.columns = [ 'Study', 'under50-bin', '50-60-bin', '60-70-bin', '70up-bin', 'Predictions' ]
        df.to_csv( output +'.csv', index=False )
    
    return mae, baseline_mae



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
    parser.add_argument('--h5_file', type=str, 
                        help='Name of the h5py file')
    parser.add_argument('--study', type=str, default='test',
                        help='Name of the experiment/complex study')
    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--get_coeff',  type=str, default='rho',
                        help='Type of error coefficient we need, MAE or Spearmanns rho. By default we will output rho')
    
    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    output = args.output
    h5_file = args.h5_file
    model_path = args.model_path
    study = args.study
    err_coeff = args.get_coeff

    model_list = [ model_path + "ensemble/Model1.torch",
                   model_path + "ensemble/Model2.torch",
                   model_path + "ensemble/Model3.torch" ]

    if( err_coeff == 'mae' ):
        evaluate_consensus_mae(h5_file, model_list, output, study)
    else:
        evaluate_consensus(h5_file, model_list, output, study)

if __name__ == '__main__':
    _cli()


