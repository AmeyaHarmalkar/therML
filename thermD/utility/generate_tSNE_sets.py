import h5py
import numpy as np
import torch
import argparse
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import tqdm
from tqdm import tqdm
from datetime import datetime
from einops import rearrange, repeat


# Model evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm.notebook import tqdm
from time import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go


import supervised
from supervised.utility.tensor import pad_data_to_same_shape
from supervised.datasets import PerResDataset
from supervised.datasets.PerResDataset import PerResidueDataset
#from supervised.networks.SmallCNNs import *



def load_dataset( h5_file, batch_size=3000 ):

    dataset = PerResidueDataset(h5_file)
    test_loader = data.DataLoader( dataset,
                                   batch_size=batch_size, shuffle=True,
                                   collate_fn=PerResidueDataset.merge_samples_to_minibatch )
    return test_loader


def get_hD_outputs(model_path, layer, test_loader, device):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model_test = torch.load(model_path, map_location='cpu')
    model_test.fc1.register_forward_hook( get_activation(layer) )
    
    for seq_inputs in tqdm(test_loader):
        #print( len(seq_inputs) )
        inputs = seq_inputs[0]
        sequence_input = inputs[0]
        enr_input = inputs[1]
        labels = seq_inputs[1].long()
        outputs = model_test(sequence_input, enr_input)
        cluster = activation[layer]
        cluster = cluster.view( cluster.size(0), -1)
        print("Cluster Shape : ", cluster.shape)
        print("Label shape : ", len(labels))
    
    return cluster, labels


def get_labeled_hd_outputs(model_path, layer, test_loader, device):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model_test = torch.load(model_path, map_location='cpu')
    model_test.fc1.register_forward_hook( get_activation(layer) )

    for seq_inputs in tqdm(test_loader):
        print(len(seq_inputs))
        inputs = seq_inputs[0]
        sequence_input = inputs[0]
        enr_input = inputs[1]
        labels = seq_inputs[1].long()
        token = seq_inputs[2]
        outputs = model_test(sequence_input, enr_input)
        cluster = activation[layer]
        cluster = cluster.view( cluster.size(0), -1)
        print("Cluster Shape : ", cluster.shape)
        print("Label shape : ", len(labels))
        print("Token shape : ", len(token))
    
    return cluster, labels, token


def run_tsne(cluster, labels):
    
    # define runtime
    start = time()
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(cluster)
    
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    print('pca complete %.2f s' % (time() - start))
    
    start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1500)
    X_pca_bh = tsne.fit_transform(cluster)
    print(pca_result.shape, X_pca_bh.shape)
    print('time elapsed %.2f s' % (time() - start))
    labels_index = labels.tolist()
    
    return X_pca_bh, labels_index


def run_labelled_tsne(cluster, labels, tokens):
    
    # define runtime
    start = time()
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform(cluster)
    
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    print('pca complete %.2f s' % (time() - start))
    
    start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=2500)
    X_pca_bh = tsne.fit_transform(cluster)
    print(pca_result.shape, X_pca_bh.shape)
    print('time elapsed %.2f s' % (time() - start))
    labels_index = labels.tolist()
    
    return X_pca_bh, labels_index, tokens


def plot_tSNE_figure(X_pca_bh, labels_index, save_file, out_dir):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_pca_bh[:, 0], y=X_pca_bh[:, 1],  mode='markers',
                            marker=dict(color=labels_index, colorscale='viridis', showscale=True,)) )
    
    fig.update_xaxes(mirror=True, linewidth=2, linecolor='black',
                        tickfont=dict(family="Helvetica", color='black',size=36))
    fig.update_yaxes(tickfont=dict(family="Helvetica", color='black',size=36),
                            mirror=True, linewidth=2, linecolor='black',showgrid=True,)
    fig.update_traces(marker=dict(size=8,line=dict(width=0.4,color='White')))
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_layout(width=600, height=600)
    fig.write_image( os.path.join(out_dir, save_file + '_temp.png') , format="png", scale=3)


def plot_labelled_tSNE_figure(X_pca_bh, labels_index, tokens, save_file, out_dir):

    color_dict = {
    'CCR8' : 0,
    'CD123': 1,
    'CD22starting': 2,
    'CD22AFFMAT': 2,
    'CLDN18.2': 3,
    'MUC1-SEA': 4,
    'CD70': 5,
    'LRRC15': 6,
    'CS1': 7,
    'CS115G8Opt': 7,
    'CS16C12Opt': 7,  
    'MAGEB2': 8,
    'PSMAAFFMAT': 9,
    'CLL1opt': 10,
    'CLL1': 10,
    'DCAF4L2': 11,
    'EpCAM': 12,
    'MUC16': 13,
    'MUC17': 14,
    'MUC13': 15,
    'CD20': 16,
    '': 17}

    project_names = {
    'CCR8' : 'Set A',
    'CD123': 'Set B',
    'CD22starting': 'Set C',
    'CD22AFFMAT': 'Set C',
    'CLDN18.2': 'Set D',
    'MUC1-SEA': 'Set E' ,
    'CD70': 'Set F',
    'LRRC15': 'Set G',
    'CS1': 'Set H',
    'CS115G8Opt': 'Set H',
    'CS16C12Opt': 'Set H',  
    'MAGEB2': 'Set I',
    'PSMAAFFMAT': 'Set J',
    'CLL1opt': 'Set K',
    'CLL1': 'Set K',
    'DCAF4L2': 'Set L',
    'EpCAM': 'Set M',
    'MUC16': 'Set N',
    'MUC17': 'Set O',
    'MUC13': 'Set P',
    'CD20': 'Set Q',
    '': 17}

    token_color_list = []
    project_list = []

    for i in range(len(tokens)):
        token_color_list.append( color_dict[tokens[i].decode()] )
        project_list.append( project_names[tokens[i].decode()] )

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_pca_bh[:, 0], y=X_pca_bh[:, 1],  mode='markers', text=project_list,
                            marker=dict(color=token_color_list, colorscale='rainbow', showscale=True,)) )
    
    fig.update_xaxes(mirror=True, linewidth=2, linecolor='black',
                        tickfont=dict(family="Helvetica", color='black',size=36))
    fig.update_yaxes(tickfont=dict(family="Helvetica", color='black',size=36),
                            mirror=True, linewidth=2, linecolor='black',showgrid=True,)
    fig.update_traces(marker=dict(size=8,line=dict(width=0.4,color='White')))
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_layout(width=600, height=600)
    fig.write_image( os.path.join(out_dir, save_file + '_exp.png') , format="png", scale=3)

    
    

def _get_args():
    """Obtain the command line arguments"""
    desc = ('''
        Script for training a pairwise sequence model with Ab sequences. 
        The H5 file required can be generated by using
        generate_h5_file.py. The default model to use is the
        CNN1D model which comprises of convolutional layers with a fully-
        connected last layer for classification.
    ''')

    parser = argparse.ArgumentParser(description=desc)
    
    #training arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path of the model to be used for tSNE')
    parser.add_argument('--save_file', type=str, default='output',
                        help='name of the model to save post last epoch')
    parser.add_argument('--out_dir', type=str, 
                        help='Destination to save file')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='Number of Ab sequences per batch')
    parser.add_argument('--try_gpu', type=bool, default=False,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--h5_file', type=str, 
                        help=('The name of the H5 file'))
    parser.add_argument('--exp_label', type=str, default=False,
                        help=('Whether the set is of experimental labels or not'))
    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    h5_file = args.h5_file
    model_path = args.model_path
    batch_size = args.batch_size
    save_file = args.save_file
    out_dir = args.out_dir
    exp_label = args.exp_label

    test_loader = load_dataset(h5_file, batch_size=batch_size)

    # Assuming that the FC layer is what we need
    if (exp_label):
        cluster, labels, tokens = get_labeled_hd_outputs(model_path, 'fc1', test_loader, device)
        X_pca_bh, labels_index, tokens = run_labelled_tsne(cluster, labels, tokens)
        plot_labelled_tSNE_figure(X_pca_bh, labels_index, tokens, save_file, out_dir)
        plot_tSNE_figure(X_pca_bh, labels_index, save_file, out_dir)
    else:
        cluster, labels = get_hD_outputs(model_path, 'fc1', test_loader, device)
        # Running tSNE
        X_pca_bh, labels_index = run_tsne(cluster, labels)
        plot_tSNE_figure(X_pca_bh, labels_index, save_file, out_dir)
    


if __name__ == '__main__':
    _cli()



    