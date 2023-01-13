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


class PerResidueDataset(Dataset):
    """This class loads in the sequence and residue-wise energy data and one-hot
    encodes it to pass it to the Dataloader.
    """

    def __init__(self, filename, onehot_prim=True):
        """Initialize the class
        ** Arguments **
            filename : (string) 
                Path of the h5py file
            onehot_data : (bool) 
                If the sequence and energy data should be one-hot encoded or not
        """
        super(PerResidueDataset, self).__init__()
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
        pairwise_energy_prim = self.h5file['pairwise_energy_mat'][index, :total_seq_len, :total_seq_len]

        # Convert everything to torch tensors
        heavy_prim = torch.Tensor(heavy_prim).type(dtype=torch.uint8)
        light_prim = torch.Tensor(light_prim).type(dtype=torch.uint8)
        
        pairwise_energy_prim = torch.Tensor(pairwise_energy_prim).type(dtype=torch.uint8)

        # obtain metadata
        metadata = self.h5file['label'][index]
        token = self.h5file['token'][index]

        return index, heavy_prim, light_prim, heavy_seq_len, light_seq_len, pairwise_energy_prim, token, metadata
    
    def __len__(self):
        return self.num_proteins


    @staticmethod
    def merge_samples_to_minibatch(samples):
        # sort according to length of aa sequence
        samples.sort(key=lambda x: len(x[2]), reverse=True)
        return PerResidueFeatureDataset(zip(*samples)).data()



class PerResidueFeatureDataset:

    def __init__(self, batch_data):
        (self.index, self.heavy_prim, self.light_prim, self.heavy_seq_len, 
        self.light_seq_len, self.pairwise_energy_prim,
        self.token, self.metadata) = batch_data
    
    def data(self):
        return self.features(), self.labels(), self.tokens(), self.seq_lengths()

    def features(self):

        seq_start, seq_end, seq_delim, data_end = torch.tensor([20]).byte(), torch.tensor([21]).byte(), torch.tensor([22]).byte(), torch.tensor([23]).byte()

        h_len = self.heavy_seq_len[0]
        l_len = self.light_seq_len[0]
        total_len = h_len + l_len

        combined_seqs = [
            torch.cat([h,l])
            #torch.cat([seq_start, h, l, seq_end])
            for h,l in zip(self.heavy_prim, self.light_prim)
        ]

        combined_seqs = pad_data_to_same_shape(combined_seqs, pad_value=20)
        combined_seqs = torch.stack( [F.one_hot(seq.long(), num_classes=21) for seq in combined_seqs])
        combined_seqs = combined_seqs.float().transpose(1, 2)

        
        combined_res = pad_data_to_same_shape(self.pairwise_energy_prim, pad_value=20)
        
        combined_res = torch.stack( [F.one_hot(seq.long(), num_classes=21) for seq in combined_res])
        combined_res = combined_res.float().transpose(1, 3)

        return combined_seqs, combined_res


    def seq_lengths(self):
        h_len = [x for x in self.heavy_seq_len]
        l_len = [x for x in self.light_seq_len]
        #total_len = h_len + l_len
        return h_len, l_len


    def labels(self):
        # to convert the masks as an array
        combined_label = [ x for x in self.metadata]
        combined_label = torch.tensor(combined_label)
        return combined_label


    def tokens(self):
        # to convert the masks as an array
        combined_token = [ x for x in self.token]
        #combined_token = torch.tensor(combined_token)
        return combined_token


def h5_ab_residues_dataloader(filename, batch_size=1, shuffle=True, **kwargs):
    constant_kwargs = ['collate_fn']
    if any([c in constant_kwargs for c in kwargs.keys()]):
        raise ValueError(
            'Cannot modify the following kwargs: {}'.format(constant_kwargs))

    kwargs.update( dict(collate_fn=PerResidueDataset.merge_samples_to_minibatch) )
    kwargs.update(dict(batch_size=batch_size))

    return data.DataLoader(PerResidueDataset(filename), **kwargs, shuffle=shuffle)






