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
from torch.optim import Adam,AdamW
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


def load_dataset( h5_file, data_split=0.8, batch_size=50 ):

    dataset = PerResidueDataset(h5_file)
    train_split_length = int( len(dataset) * data_split )

    train_dataset, test_dataset = data.random_split(dataset, [train_split_length, len(dataset) - train_split_length])
    train_loader = data.DataLoader( train_dataset, batch_size=batch_size, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )
    validation_loader = data.DataLoader( test_dataset, batch_size=batch_size, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )

    return train_loader, validation_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def train_epoch(model, train_loader, optimizer, device, criterion):
    """Train model for one epoch"""

    model.train()
    running_losses = 0

    for input_data in tqdm(train_loader):

        inputs = input_data[0]
        sequence_input = inputs[0].to(device)
        enr_input = inputs[1].to(device)
        labels = input_data[1].long().to(device)


        optimizer.zero_grad()

        def handle_batch():
            outputs = model(sequence_input, enr_input)
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
            sequence_input = inputs[0].to(device)
            enr_input = inputs[1].to(device)
            labels = input_data[1].long().to(device)

            def handle_batch():
                outputs = model(sequence_input, enr_input)
                loss = criterion(outputs, labels)
                return outputs, loss.item()
            
            outputs, batch_loss = handle_batch()
            validation_loss += batch_loss
        
    return validation_loss


def train(model, train_loader, validation_loader, optimizer, epochs, 
          device, criterion, lr_modifier, save_every, out_dir, save_file='output'):
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

        def plot_loss(train_losses, validation_losses, out_dir):

            plt.figure(dpi=500)
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams.update({'font.size': 20})
            plt.plot(train_losses, label="Train")
            plt.plot(validation_losses, label="Validation")
            plt.ylabel("CCE Loss")
            plt.xlabel("Epoch")
            plt.legend()
            plt.savefig(os.path.join(out_dir, "Loss.png"))
            plt.close()
            np.savetxt(os.path.join(out_dir, "loss_data.csv"),
                    np.array([
                    list(range(len(train_losses))), train_losses,
                    validation_losses
                    ]).T,
                delimiter=",")
        
        plot_loss(train_losses, validation_losses, out_dir)

        if( ( epoch + 1 ) % save_every == 0 ):
            torch.save(model, os.path.join( out_dir, save_file + ".e{}".format( epoch + 1 )) )
        
    torch.save(model, os.path.join( out_dir, save_file + ".torch" ) )


def plot_cm(y_true, y_pred, labels=None, figsize=(10,10), outfile='output.png'):
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


def evaluate_testdata( dataloader, model_path, out_path, device):

    test_loader = dataloader
    
    model_test = torch.load(model_path)
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
            model_preds.append( model_test(sequence_input, enr_input).argmax(1) )
            final_labels.append( labels  )

    final_labels = torch.cat(final_labels)
    model_preds = torch.cat(model_preds)

    model_names = ["CNN-Model"]
    model_predictions = [model_preds]

    res = np.concatenate([
        np.array([
            "Label", "CNN-Model"])[None, :],
        torch.stack([
            final_labels.cpu(), model_preds.cpu()]).numpy().T
    ])

    for i in range(len(model_names)):
        plot_cm(   model_predictions[i], final_labels, [ '0-50', '50-60', '60-70', '70 up' ], outfile=out_path)


def evaluate_designset( dataloader, model_path, out_path, device):

    test_loader = dataloader
    
    model_test = model_path.to(device)
    model_test.eval()

    final_labels = []
    model_preds = []

    with torch.no_grad():
        for seq_inputs in tqdm(test_loader):
            inputs = seq_inputs[0]
            sequence_input = inputs[0].to(device)
            enr_input = inputs[1].unsqueeze(1).to(device)
            labels = seq_inputs[1].long().to(device)
            model_preds.append( model_test(sequence_input, enr_input).argmax(1) )
            final_labels.append( labels  )

    final_labels = torch.cat(final_labels)
    model_preds = torch.cat(model_preds)

    model_names = ["CNN-Model"]
    model_predictions = [model_preds]

    res = np.concatenate([
        np.array([
            "Label", "CNN-Model"])[None, :],
        torch.stack([
            final_labels.cpu(), model_preds.cpu()]).numpy().T
    ])

    for i in range(len(model_names)):
        plot_cm(   model_predictions[i], model_predictions[i], [ '0-50', '50-60', '60-70', '70 up' ], outfile=out_path)


def run_design_datasets(model, out_path, device):

    test_data = { "2B12-H" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Design-data/residue_2B12-H_data.h5'),
                  "2B12-L" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Design-data/residue_2B12-L_data.h5'),
                  "CD40-H" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Design-data/residue_CD40-H_data.h5'),
                  "CD40-L" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Design-data/residue_CD40-L_data.h5'),
    }

    for k,v in test_data.items():

        test_loader = data.DataLoader( v, batch_size=32, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )
        evaluate_designset( test_loader, model, out_path + k + ".png", device )


def evaluate_models(model, validation_loader, device):
    model = model.to(device)
    model.eval()
    
    final_labels = []
    model_preds = []

    
    with torch.no_grad():
        for seq_inputs in tqdm(validation_loader):
            inputs = seq_inputs[0]
            sequence_input = inputs[0].to(device)
            enr_input = inputs[1].to(device)
            labels = seq_inputs[1].long().to(device)
            model_preds.append( model(sequence_input, enr_input).cpu().argmax(1) )
            final_labels.append( labels.cpu() )

    final_labels = torch.cat(final_labels)
    model_preds = torch.cat(model_preds)
    
    mcc_score = matthews_corrcoef(final_labels, model_preds)
    accuracy = accuracy_score(final_labels, model_preds)
    print("Accuracy : ", accuracy)
    print("MCC : ", mcc_score)
    
    return accuracy, mcc_score


def evaluate_spearmann_corr( test_loader, model, device):
    """This function is specifically to calculate the spearmann's
    correlation coefficient based on the TS50 dataset. 
    """

    model = model.to(device)
    model.eval()
    
    def predict_ts50(value):
        """The mean_ts50 values are determined from the average TS50 value 
        of sequences in the TS50 dataset in each of the four bins i.e 
        [under 50, 50-60, 60-70, 70up]. We compute probabilities in each
        bin with our model and then equate the spearmann correlation coeff
        between this expected value and the actual value."""
        mean_ts50 = torch.tensor([36.0395, 56.6874, 64.190, 70.0740])
        expected_value = torch.dot(value, mean_ts50)
        return expected_value.numpy()
    
    tokens = []
    model_preds = []
    a = 0
    with torch.no_grad():
        for seq_inputs in tqdm(test_loader):
            inputs = seq_inputs[0]
            sequence_input = inputs[0].to(device)
            enr_input = inputs[1].to(device)
            labels = seq_inputs[1].long().to(device)
            token = seq_inputs[2]
            value = F.softmax(model(sequence_input, enr_input), -1)
            model_preds.append( value.cpu() )
            tokens.append(token)
    
    tokens = list(chain(*tokens))
    model_preds = torch.cat(model_preds)
    
    ts50_preds = []
    data_list = []
    
    for i in range(len(tokens)):
        exp = predict_ts50(model_preds[i])
        ts50_preds.append(exp)
        data_list.append( [tokens[i], exp] )
        
    data_list = list(map(list,zip(*data_list)))
    df = pd.DataFrame(np.transpose(data_list), columns = [ 'Actual', 'Prediction' ] )
    rho = df[["Actual", "Prediction"]].corr(method="spearman").iloc[0::2,1::2].iloc[0,0].item()
    print("Spearmann corr : ", rho)
    return rho



def run_test_datasets(model, data_list, device):

    test_data = { "I2C" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Annotated/pairwise-I2C.h5'),
                  "isoVH" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Annotated/pairwise-isoVH.h5'),
                  "MUC13" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Annotated/pairwise-MUC13-MUC16.h5'),
                  "CD20" : PerResidueDataset('/home/aharmal1/ThermDesign/data/train_data/ResidueWise/H5PY/Annotated/pairwise-CD20.h5')
    }

    for k,v in test_data.items():

        test_loader = data.DataLoader( v, batch_size=32, shuffle=True, collate_fn=PerResidueDataset.merge_samples_to_minibatch )
        acc, corr = evaluate_models(model, test_loader, device)
        rho = evaluate_spearmann_corr( test_loader, model, device )
        data_list.append([k, acc, corr, rho])
    
    return data_list


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
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save model every X number of epochs.')
    parser.add_argument('--save_file', type=str, default='output',
                        help='name of the model to save post last epoch')
    parser.add_argument('--out_dir', type=str, 
                        help='Destination to save file')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of Ab sequences per batch')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for Adam')
    parser.add_argument('--try_gpu', type=bool, default=True,
                        help='Whether or not to check for/use the GPU')
    parser.add_argument('--train_split', type=float, default=0.7,
                        help=('The percentage of the dataset that is used '
                              'during training'))
    parser.add_argument('--h5_file', type=str, 
                        help=('The name of the H5 file'))
    parser.add_argument('--ignore_seq', type=bool, default=False,
                        help=('Whether to ignore sequences while training'))
    parser.add_argument('--ignore_energy', type=bool, default=False, 
                        help=('Whether to ignore energy features while training'))
    parser.add_argument('--num_1d_blocks',type=int, default=1,
                        help=('Number of 1D blocks in the sequence branch'))
    parser.add_argument('--num_2d_blocks',type=int, default=2,
                        help=('Number of 2D blocks in the sequence branch'))
    return parser.parse_args()


def _cli():
    args = _get_args()

    device_type = 'cuda' if torch.cuda.is_available() and args.try_gpu else 'cpu'
    device = torch.device(device_type)

    h5_file = args.h5_file
    data_split = args.train_split
    batch_size = args.batch_size
    save_file = args.save_file
    out_dir = args.out_dir
    ignore_seq = args.ignore_seq
    ignore_energy = args.ignore_energy
    num_1d_blocks = args.num_1d_blocks
    num_2d_blocks = args.num_2d_blocks

    lr = args.lr

    train_loader, validation_loader = load_dataset(h5_file, data_split, batch_size)

    model = EquiCNN(ignore_seq=ignore_seq,ignore_energy=ignore_energy,num_1d_blocks=1,num_2d_blocks=1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_modifier = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.CrossEntropyLoss()

    print('Arguments:\n', args)
    print('Model:\n', model)
    print('\n Conv model : {} parameters to fit'.format(count_parameters(model)))

    train(model=model,
          train_loader=train_loader,
          validation_loader=validation_loader,
          optimizer=optimizer,
          device=device,
          epochs=args.epochs,
          criterion=criterion,
          lr_modifier=lr_modifier,
          save_every=args.save_every,
          save_file=save_file,
          out_dir=out_dir)

    # Load the model and plot the confusion matrix
    model_path = os.path.join( out_dir, save_file + ".torch" )
    out_path = os.path.join( out_dir, 'validation.png' )

    data_dict = []
    model_trained = torch.load(model_path)
    model_trained = model_trained.to(device)
    acc1, corr1 = evaluate_models(model_trained, validation_loader, device)
    data_dict.append([ "Energy_sequence", acc1, corr1 ])
    run_test_datasets(model_trained, data_dict, device)
    data_list = list(map( list, zip(*data_dict)))
    df = pd.DataFrame(data_list).transpose()
    df.to_csv(out_dir+"/EquiCNN_performance.txt", index=False)

    evaluate_testdata( train_loader, model_path, os.path.join( out_dir, 'trained.png' ), device) 
    evaluate_testdata( validation_loader, model_path, os.path.join( out_dir, 'validation.png' ), device) 
    #run_design_datasets(model_trained, out_dir, device)


if __name__ == '__main__':
    _cli()
