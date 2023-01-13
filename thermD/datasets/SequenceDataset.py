## Code derived from RosettaCommons/DeepAb
import h5py
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from thermD.utility.tensor import pad_data_to_same_shape


class PairedSequenceDataset(Dataset):

    def __init__(self, filename, onehot_prim=True):
        """Initialize the class
        ** Arguments **
            csv_file : (string) 
                Path of the csv file
            onehot_data : (bool) 
                If the sequence data should be one-hot encoded or not
        """
        super(PairedSequenceDataset, self).__init__()
        self.onehot_prim = onehot_prim
        self.filename = filename
        self.h5file = h5py.File(filename, 'r')
        self.num_proteins, _ = self.h5file['heavy_chain_primary'].shape
    

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise IndexError('Slicing not supported')

        heavy_seq_len = self.h5file['heavy_chain_seq_len'][index]
        light_seq_len = self.h5file['light_chain_seq_len'][index]
        total_seq_len = heavy_seq_len + light_seq_len

        # Obtain the attributes from the protein and cut off zero padding
        heavy_prim = self.h5file['heavy_chain_primary'][index, :heavy_seq_len]
        light_prim = self.h5file['light_chain_primary'][index, :light_seq_len]

        # Convert to torch tensors
        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)

        # obtain metadata
        metadata = self.h5file['label'][index]

        return index, heavy_prim, light_prim, metadata

    
    def __len__(self):
        return self.num_proteins

    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return SequenceDataset(zip(*samples)).data()        

    
class SequenceDataset:
    """This class handles each protein sequence that is loaded by the PairedSequenceDataset
    Class. It adds start, end and break points in the sequence and outputs a one-hot
    encoded tensor via the features method.
    """
    def __init__(self, batch_data):
        (self.index, self.heavy_prim, self.light_prim, self.metadata) = batch_data

    def data(self):
        return self.features(), self.masks()

    def features(self):
        seq_start, seq_end, seq_delim = torch.tensor([20]).byte(), torch.tensor([21]).byte(), torch.tensor([22]).byte()
        
        combined_seqs = [
            torch.cat([seq_start, h, seq_delim, l, seq_end])
            for h, l in zip(self.heavy_prim, self.light_prim)
        ]

        combined_seqs = pad_data_to_same_shape(combined_seqs, pad_value=22)

        combined_seqs = torch.stack(
            [F.one_hot(seq.long()) for seq in combined_seqs])

        combined_seqs = combined_seqs.float().transpose(1, 2)

        return combined_seqs

    def labels(self):
        seq_start, seq_end, seq_delim = torch.tensor(
            [20]).byte(), torch.tensor([21]).byte(), torch.tensor([22]).byte()

        combined_seqs = [
            torch.cat([seq_start, h, seq_delim, l, seq_end])
            for h, l in zip(self.heavy_prim, self.light_prim)
        ]
        combined_seqs = pad_data_to_same_shape(combined_seqs, pad_value=22)

        return combined_seqs.long()

    def masks(self):
        # to convert the masks as an array
        combined_mask = [ x for x in self.metadata]
        combined_mask = torch.tensor(combined_mask)
        return combined_mask


def h5_antibody_dataloader(filename, batch_size=1, shuffle=True, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update( dict(collate_fn=PairedSequenceDataset.merge_samples_to_minibatch) )
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(PairedSequenceDataset(filename), **kwargs, shuffle=shuffle)






