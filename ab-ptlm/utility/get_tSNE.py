import h5py
import numpy as np
import torch
import argparse
from pathlib import Path
import subprocess
import pickle as pkl
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


from evo.parsing import read_sequences
from evo.sequence import create_mutant_df
from inference import SupervisedAntiBERTyScorer
from dataset import split_linkers


parser = argparse.ArgumentParser()
parser.add_argument( "filename", type=Path, help="CSV file with the sequences, temperature and experiment data" )
parser.add_argument(
    "--model", 
    choices=["antiberty_supervised"], # Need to add the unsupervised model here
    default="antiberty_supervised",
    required=True
)
parser.add_argument(
    '--save_file', type=str, 
    default='output',
    help='name of the model to save post last epoch'
)
parser.add_argument(
    '--out_dir', type=str, 
    default='Results/',
    help='Destination to save file'
)
args = parser.parse_args()


def read_csvfile(filename):

    df = pd.read_csv(filename, headers=0)
    tokens = [ x for x in df["Group"]]
    labels = [ x for x in df["label"]]
    sequences = []
    for seq in df["Sequence"]:
        heavy, light = split_linkers(seq)
        sequences.append( {'H': heavy, 'L': light }  )
    return tokens, labels, sequences


def get_tSNE_inputs( filename, layer='linear' ):

    model = SupervisedAntiBERTyScorer()
    test_model = model.summary()

    df = pd.read_csv( filename, header = 0 )
    tokens = [ x for x in df["Group"]]
    labels = [ int(x) for x in df["label"]]
    for i in range(len(labels)):
        if (labels[i] < 0):
            print( i, labels[i], tokens[i])
    

    cluster = [ ]
    for seq in df["Sequence"]:
        seq_list = []
        heavy, light = split_linkers(seq)
        seq_list.append( {'H': heavy, 'L': light } )

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        h1 = test_model.output.mlp[0].register_forward_hook( get_activation('linear') )
        h2 = test_model.output.mlp[1].register_forward_hook( get_activation('tanh') )
        h3 = test_model.output.mlp[2].register_forward_hook( get_activation('dropout') )
        h4 = test_model.output.mlp[3].register_forward_hook( get_activation('linear_class') )

        # Forward pass to get the output
        output = model.score(seq_list)
        cluster.append( torch.squeeze(activation[layer]).cpu().detach().numpy() )

        h1.remove()
        h2.remove()
        h3.remove()
        h4.remove()
    
    return cluster, labels, tokens


def run_tsne( cluster, labels, tokens ):

    start = time()
    pca = PCA(n_components=4)
    pca_result = pca.fit_transform( cluster )
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca.explained_variance_ratio_)))
    print('pca complete %.2f s' % (time() - start))
    
    start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1500)
    X_pca_bh = tsne.fit_transform(cluster)
    print(pca_result.shape, X_pca_bh.shape)
    print('time elapsed %.2f s' % (time() - start))
    
    return X_pca_bh, labels, tokens


def plot_tSNE_figure(X_pca_bh, labels_index, tokens, save_file, out_dir):

    color_dict = { 
        'CCR8' : 0,
        'CD123': 1,
        'CD22' : 2,
        'CLDN18.2': 3,
        'MUC1-SEA': 4,
        'CD70': 5,
        'LRRC15': 6,
        'CS1': 7,
        'MAGEB2': 8,
        'PSMA': 9,
        'CLL1': 10,
        'DCAF4L2': 11,
        'EpCAM': 12,
        'MUC16': 13,
        'MUC17': 14,
        'MUC13': 15,
        'CD20': 16,
        '': 17
    }

    temp_names = {
        0 : 'under 50',
        1 : '50-60',
        2 : '60-70',      
        3 : '70 up'
    }

    project_names = {
        'CCR8' : 'Set A',
        'CD123': 'Set B',
        'CD22': 'Set C',
        'CLDN18.2': 'Set D',
        'MUC1-SEA': 'Set E' ,
        'CD70': 'Set F',
        'LRRC15': 'Set G',
        'CS1': 'Set H',
        'MAGEB2': 'Set I',
        'PSMA': 'Set J',
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
        token_color_list.append( color_dict[tokens[i]] )
        project_list.append( project_names[tokens[i]] )
    colormap_tokens = 'rainbow'
    colormap_labels = 'viridis'

    for i in range(len(labels_index)):
        if (labels_index[i] < 0 ):
            print(labels_index[i] )

    print( "Plotting tSNE with study labels" )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_pca_bh[:, 0], y=X_pca_bh[:, 1],  mode='markers', text=project_list,
                            marker=dict(color=token_color_list, colorscale=colormap_tokens, showscale=True,)) )
    fig.update_xaxes(mirror=True, linewidth=2, linecolor='black',
                        tickfont=dict(family="Helvetica", color='black',size=36))
    fig.update_yaxes(tickfont=dict(family="Helvetica", color='black',size=36),
                            mirror=True, linewidth=2, linecolor='black',showgrid=True,)
    fig.update_traces(marker=dict(size=8,line=dict(width=0.4,color='White')))
    fig.update(layout_coloraxis_showscale=True)
    fig.update_layout(plot_bgcolor='white')
    fig.update_layout(yaxis_zeroline=True, xaxis_zeroline=True)
    fig.update_layout(width=600, height=600)
    fig.write_image( os.path.join(out_dir, save_file + '_tokens.png') , format="png", scale=3)

    print( "Plotting tSNE with temperature labels" )
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X_pca_bh[:, 0], y=X_pca_bh[:, 1],  mode='markers',
                            marker=dict(color=labels_index, colorscale=colormap_labels, showscale=True,)) )
    fig1.update_xaxes(mirror=True, linewidth=2, linecolor='black',
                        tickfont=dict(family="Helvetica", color='black',size=36))
    fig1.update_yaxes(tickfont=dict(family="Helvetica", color='black',size=36),
                            mirror=True, linewidth=2, linecolor='black',showgrid=True,)
    fig1.update_traces(marker=dict(size=8,line=dict(width=0.4,color='White')))
    fig1.update(layout_coloraxis_showscale=True)
    fig1.update_layout(plot_bgcolor='white')
    fig1.update_layout(yaxis_zeroline=True, xaxis_zeroline=True)
    fig1.update_layout(width=600, height=600)
    fig1.write_image( os.path.join(out_dir, save_file + '_labels.png') , format="png", scale=3)



layers = [ 'linear', 'tanh', 'dropout', 'linear_class']
for layer in layers:
    cluster, labels, tokens = get_tSNE_inputs( args.filename, layer )
    X_pca_bh, labels_index, tokens = run_tsne(cluster, labels, tokens)
    plot_tSNE_figure(X_pca_bh, labels_index, tokens, layer + '_' + args.save_file, args.out_dir)