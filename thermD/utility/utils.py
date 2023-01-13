import torch
import torch.nn.functional as F
import re
import argparse
import pandas as pd
import numpy as np
from os.path import splitext, basename
from Bio import SeqIO


_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19'
}

_aa_3letter_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP'
}

_rev_aa_dict = {v: k for k, v in _aa_dict.items()}


def letter_to_num(string, dict_):
    """ Convert string of letters to list of ints 
    Function obtained from ProteinNet (https://github.com/aqlaboratory/proteinnet/blob/master/code/text_parser.py)
    """
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def one_hot_seq(seq):
    """Obtain a one-hot encoding of protein sequence
    """
    return F.one_hot( torch.LongTensor(letter_to_num( seq, _aa_dict )), num_classes=20 )


def get_fasta_chain_seq(fasta_file, chain_id):
    """Obtain the sequence corresponding to the chain ID"""
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ":{}".format(chain_id) in chain.id:
            return str(chain.seq)


def get_heavy_seq_len(fasta_file):
    h_len = len(get_fasta_chain_seq(fasta_file, "H"))
    return h_len


def get_light_seq_len(fasta_file):
    l_len = len(get_fasta_chain_seq(fasta_file, "L"))
    return l_len

    
def lev_distance(s1, s2):
    oh1 = one_hot_seq(s1)
    oh2 = one_hot_seq(s2)
    return torch.sum((oh1 - oh2) > 0).item()


def get_id(pdb_file_path):
    return splitext(basename(pdb_file_path))[0]

def get_energy_id(energy_file_path):
    outfile = splitext(basename(energy_file_path))[0]
    if outfile.startswith('CLDN18'):
        out = outfile.split('_msa_CLDN18_')

    else:
        out = outfile.split('_msa_')
    return out[0], out[1]


def protein_residue_energy(energy_file, lend=-25, uend=10, nbins=20, device=None):
    """This function parses the residue energy breakdown output, flattens it
    in 1-D and generates a torch tensor to pass on ahead. The data is binned 
    in 20 bins between lower-end and upper-end of (-25,10).
    """
    energy = pd.read_csv(energy_file, header=0, delim_whitespace=1)
    max_len = int( energy['resi1'].max() )
    energy_data = np.zeros(max_len, )
    
    for res1, pdb1, res2, pdb2, total in zip( 
        energy['resi1'], energy['pdbid1'], energy['resi2'], energy['pdbid2'], energy['total'] ):
        
        if (res2 == '--'):
            key = int(res1) - 1
            energy_data[key] += total
        else:
            res1_key = int(res1) - 1
            res2_key = int(res2) - 1
            energy_data[res1_key] += total
            energy_data[res2_key] += total
    
    bins = np.linspace(lend, uend, num=nbins)
    data = np.digitize(energy_data, bins)
    
    return data


def protein_pairwise_energy_matrix(energy_file, lend=-5, uend=5, nbins=20, device=None):
    """This function parses the residue energy breakdown output 
    and generates a torch tensor to pass on ahead. The data is binned
    in 20 labels between (-5,5).
    """
    energy = pd.read_csv(energy_file, header=0, delim_whitespace=1)
    max_len = int( energy['resi1'].max() )
    energy_data = np.zeros((max_len, max_len))
    
    for res1, pdb1, res2, pdb2, total in zip( 
        energy['resi1'], energy['pdbid1'], energy['resi2'], energy['pdbid2'], energy['total'] ):
        
        if (res2 == '--'):
            key = int(res1) - 1
            energy_data[key][key] = total
        else:
            res1_key = int(res1) - 1
            res2_key = int(res2) - 1
            energy_data[res1_key][res2_key] = total
            energy_data[res2_key][res1_key] = total
    
    #energy_data = torch.from_numpy( energy_data )
    #data = energy_data.unsqueeze(0)

    bins = np.linspace(lend, uend, num=nbins)
    data = np.digitize(energy_data, bins)
    
    return data


def protein_pairwise_energy_data(energy_file, lend=-10, uend=10, nbins=20, device=None):
    """This function parses the residue energy breakdown output 
    and generates a torch tensor to pass on ahead. The data is binned
    in 20 labels between (-5,5).
    """
    energy = pd.read_csv(energy_file, header=0, delim_whitespace=1)
    max_len = int( energy['resi1'].max() )
    energy_data = np.zeros((max_len, max_len))
    
    for res1, pdb1, res2, pdb2, total in zip( 
        energy['resi1'], energy['pdbid1'], energy['resi2'], energy['pdbid2'], energy['total'] ):
        
        if (res2 == '--'):
            key = int(res1) - 1
            energy_data[key][key] = total
        else:
            res1_key = int(res1) - 1
            res2_key = int(res2) - 1
            energy_data[res1_key][res2_key] = total
            energy_data[res2_key][res1_key] = total
    
    #energy_data = torch.from_numpy( energy_data )
    #data = energy_data.unsqueeze(0)

    #bins = np.linspace(lend, uend, num=nbins)
    #data = np.digitize(energy_data, bins)
    
    return energy_data