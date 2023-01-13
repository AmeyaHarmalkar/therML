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
import numpy as np
import h5py
import tqdm
from tqdm import tqdm
from datetime import datetime
from thermD.networks.CNN1D import *
from thermD.preprocessing.generate_h5_file import sequences_to_h5

from thermD.datasets import MatrixDataset
from thermD.datasets.MatrixDataset import PairedMatrixDataset
from thermD.networks.ResidueNet import *


def load_dataset( h5_file, data_split=0.8, batch_size=32 ):

    dataset = PairedMatrixDataset(h5_file)
    train_split_length = int( len(dataset) * data_split )

    train_dataset, test_dataset = data.random_split(dataset, [train_split_length, len(dataset) - train_split_length])
    train_loader = data.DataLoader( train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PairedMatrixDataset.merge_samples_to_minibatch )
    validation_loader = data.DataLoader( test_dataset, batch_size=batch_size, shuffle=True, collate_fn=PairedMatrixDataset.merge_samples_to_minibatch )

    return train_loader, validation_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    


def train_epoch(model, train_loader, optimizer, device, criterion):
    """Train model for one epoch"""

    model.train()
    running_losses = 0

    for input_data in tqdm(train_loader):

        inputs = input_data[0]
        labels = input_data[1].long().to(device)

        sequence_input = inputs[0].to(device)
        energy_input = inputs[1].to(device)

        optimizer.zero_grad()

        def handle_batch():
            outputs = model(sequence_input, energy_input)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            return outputs, loss.item()
        
        outputs, batch_loss = handle_batch()
        running_losses += batch_loss
    
    return running_losses


def validate(model, validation_loader, device, criterion):
    """Validation per epoch"""
    validation_loss = 0
    with torch.no_grad():
        model.eval()
        for input_data in tqdm(validation_loader):
            inputs = input_data[0]
            labels = input_data[1].long().to(device)

            sequence_input = inputs[0].to(device)
            energy_input = inputs[1].to(device)

            def handle_batch():
                outputs = model(sequence_input, energy_input)
                loss = criterion(outputs, labels)
                return outputs, loss.item()
            
            outputs, batch_loss = handle_batch()
            validation_loss += batch_loss
        
    return validation_loss


def train(model, train_loader, validation_loader, optimizer, epochs, 
          device, criterion, lr_modifier, save_every, save_file='output'):
    """Complete train function"""
    print('Using {} as device'.format(str(device).upper()))
    model = model.to(device)

    train_losses = []
    validation_losses = []

    for epoch in range(epochs):

        train_epoch_loss = train_epoch(model, train_loader, optimizer, device, criterion)
        train_losses.append( train_epoch_loss / len(train_loader) )
        val_epoch_loss = validate(model, validation_loader, device, criterion)
        validation_losses.append(val_epoch_loss / len(validation_loader) )

        lr_modifier.step(val_epoch_loss / len(validation_loader))

        print("Train loss : ", train_losses[-1])
        print("Validation loss : ", validation_losses[-1])

        plt.figure(dpi=500)
        plt.plot(train_losses, label="Train")
        plt.plot(validation_losses, label="Validation")
        plt.ylabel("CCE Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig(os.path.join('data/', "loss.png"))
        plt.close()
        np.savetxt(os.path.join('data/', "mat_loss_data.csv"),
                   np.array([
                   list(range(len(train_losses))), train_losses,
                   validation_losses
                ]).T,
               delimiter=",")


        if( ( epoch + 1 ) % save_every == 0 ):
            torch.save(model, save_file + ".e{}".format( epoch + 1 ))
        
    torch.save(model, save_file + ".torch" )


def _get_args():
    """Obtain the command line arguments"""
    desc = ('''
        Script for training a pairwise matrix model with Ab sequences
        and per-residue energy data. The H5 file required can be generated
        by using generate_matrix_file.py. The default model to use is the
        ResidueNet model which comprises of convolutional layers with a fully-
        connected last layer for classification.
    ''')

    parser = argparse.ArgumentParser(description=desc)
    
    #training arguments
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save model every X number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Number of Ab sequences per batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam')
    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help=('The percentage of the dataset that is used '
                              'during training'))
    parser.add_argument('--h5_file', type=str, 
                        help=('The name of the H5 file'))
    parser.add_argument('--ignore_seq', type=str, default=False,
                        help=('Whether to ignore the sequence branch while training'))
    parser.add_argument('--ignore_energy', type=str, default=False,
                        help=('Whether to ignore the energy branch while training'))
    
    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    h5_file = args.h5_file
    data_split = args.train_split
    batch_size = args.batch_size
    ignore_seq = args.ignore_seq
    ignore_energy = args.ignore_energy
    lr = args.lr

    train_loader, validation_loader = load_dataset(h5_file, data_split, batch_size)

    model = ResidueCNN(ignore_seq=ignore_seq, ignore_energy=ignore_energy)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_modifier = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.CrossEntropyLoss()

    print('Arguments:\n', args)
    print('Model:\n', model)
    print('\n ResidueCNN model : {} parameters to fit'.format(count_parameters(model)))

    train(model=model,
          train_loader=train_loader,
          validation_loader=validation_loader,
          optimizer=optimizer,
          device=device,
          epochs=args.epochs,
          criterion=criterion,
          lr_modifier=lr_modifier,
          save_every=args.save_every)

if __name__ == '__main__':
    _cli()








    


