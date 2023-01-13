# Import the libraries
import pandas as pd
import numpy as np

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
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing and machine learning
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from numpy import interp
from itertools import cycle
#matplotlib.use('Qt5Agg')
import plotly.graph_objects as go


import supervised
from supervised.utility.tensor import pad_data_to_same_shape
from supervised.datasets import PerResDataset
from supervised.datasets.PerResDataset import PerResidueDataset
from supervised.networks.MiniCNNs import *
from supervised.preprocessing.generate_h5_file import sequences_to_h5


def load_dataset( h5_file, batch_size=32 ):

    dataset = PerResidueDataset(h5_file)
    test_loader = data.DataLoader( dataset,
                                   batch_size=batch_size, 
                                   collate_fn=PerResidueDataset.merge_samples_to_minibatch )
    return test_loader


def plot_cm(y_true, y_pred, labels=None, figsize=(10,10), outfile='cm_output.png'):
    cm = confusion_matrix(y_true.detach().cpu(), y_pred.detach().cpu(), labels=np.unique(y_pred.detach().cpu()))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p,c,s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p,c)
    
    if labels == None:
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    else:
        cm = pd.DataFrame(cm, index=np.unique(labels), columns=np.unique(labels))
    cm.index.name = 'Predicted'
    cm.columns.name = 'Actual'
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=2.5)
    sns.heatmap(cm, cmap= "Reds", linewidth = 0.5, annot=annot, fmt='', ax=ax,annot_kws={"size": 18})
    fig.savefig(outfile, dpi=400)


def evaluate_testdata( dataloader, model_path, device):

    test_loader = dataloader
    
    model_test = torch.load(model_path, map_location='cpu')
    model_test = model_test.to(device)
    model_test.eval()

    final_labels = []
    model_preds = []

    with torch.no_grad():
        for seq_inputs in tqdm(test_loader):
            inputs = seq_inputs[0]
            sequence_input = inputs[0].to(device)
            enr_input = inputs[1].to(device)
            labels = seq_inputs[1].long().to(device)
            #model_preds.append( model_test(sequence_input, enr_input).argmax(1) )
            model_preds.append( F.softmax( model_test(sequence_input, enr_input), -1 ) )
            final_labels.append( labels  )

    final_labels = torch.cat(final_labels)
    model_preds = torch.cat(model_preds)

    return final_labels.numpy(), model_preds.numpy()


def get_roc_data(labels, predictions):
    
    final_labels = label_binarize(labels, classes=[0,1,2,3])
    n_classes = final_labels.shape[1]
    
    # Let's compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(final_labels[:, i], predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    return fpr, tpr, roc_auc


def plot_roc_figures( test_data, model_path, out_dir, device ):
    """Plots the ROC curve only for the top-most bin
    """

    fpr_all = dict()
    tpr_all = dict()
    auc_all = dict()

    for k,v in test_data.items():
        test_loader = data.DataLoader( v, batch_size=32, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )
        labels, predictions = evaluate_testdata( test_loader, model_path, device)

        fpr_out,tpr_out,auc_out = get_roc_data(labels, predictions)
        fpr_all[k] = fpr_out
        tpr_all[k] = tpr_out
        auc_all[k] = auc_out

    fig = go.Figure()

    colors = { "I2C" : 'rgb(128,177,211)',
           "MUC13" : 'rgb(56,166,165)',
           "CD20" : 'rgb(179,222,88)',#'rgb(190,186,218)'
           "isoVH" : 'rgb(207,28,144)'
    }

    names = { "I2C" : 'Test Ab',
           "MUC13" : 'Set P',
           "CD20" : 'Set Q',
           "isoVH" : 'Isolated scFv'
    }

    fig.add_shape(type='line', line=dict(dash='dash'),
        x0=0, x1=2, y0=0, y1=2)

    for k,v in fpr_all.items():
        if(k == "isoVH" ):
            fig.add_trace( go.Scatter( x = fpr_all[k][2], y = tpr_all[k][2], mode='lines', 
                                name=names[k]+f" (AUC={auc_all[k][2]:.2f})", line=dict(
                                    color=colors[k], width=3.5) ))
        else:
            fig.add_trace( go.Scatter( x = fpr_all[k][3], y = tpr_all[k][3], mode='lines', 
                                name=names[k]+f" (AUC={auc_all[k][3]:.2f})",line=dict(
                                    color=colors[k], width=3.5) ))

    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridwidth=0.75, gridcolor='Gray')
    fig.update_yaxes(showgrid=True, gridwidth=0.75, gridcolor='Gray')

    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", 
                    font=dict(family="Helvetica",
                        size=18), showlegend=True)
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_xaxes(showgrid=True, gridwidth=0.75, gridcolor='Gray')
    fig.update_yaxes(showgrid=True, gridwidth=0.75, gridcolor='Gray')

    fig.update_xaxes(range=[0,1],mirror=True, linewidth=2, linecolor='black',
                        tickfont=dict(family="Helvetica", color='black',size=26))
    fig.update_yaxes(range=[0,1.05],tickfont=dict(family="Helvetica", color='black',size=26),
                            mirror=True, linewidth=2, linecolor='black')
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.write_image(out_dir + '_roc.png', format="png", width = 750, height=500, scale =3)
    #fig.show(show_legend=True)

    

def _get_args():
    """Obtain the command line arguments"""
    desc = ('''
        Script for obtain receiver-operator characteristics curve.
    ''')

    parser = argparse.ArgumentParser(description=desc)
    
    #training arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path of the model to be used for tSNE')
    parser.add_argument('--testset_path', type=str, default=None,
                        help='Path of the testsets')
    parser.add_argument('--out_dir', type=str, 
                        help='Destination to save file')
    parser.add_argument('--try_gpu', type=bool, default=False,
                        help='Whether or not to check for/use the GPU')
    return parser.parse_args()


def _cli():

    args = _get_args()
    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    model_path = args.model_path
    testset_path = args.testset_path
    out_dir = args.out_dir

    test_data = { "I2C" : PerResidueDataset( testset_path + 'training/I2C.h5'),
                  "isoVH" : PerResidueDataset(testset_path + 'training/isoVH.h5'),
                  "MUC13" : PerResidueDataset(testset_path + 'training/All-sets/setP.h5'),
                  "CD20" : PerResidueDataset(testset_path + 'training/All-sets/setQ.h5'),}

    
    plot_roc_figures( test_data, model_path, out_dir, device )


if __name__ == '__main__':
    _cli()



    