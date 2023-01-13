import h5py
import numpy as np
import argparse
import os
from glob import glob
import pandas as pd
import json
import itertools
import thermD
from thermD.utility.utils import _aa_dict, letter_to_num


def process_csv_data( csv_file ):
    
    col_names = pd.read_csv(csv_file, nrows=0).columns
    data = pd.read_csv(csv_file, skiprows=1, names=col_names, header=None)
    data = data[ ['Study','Name','Label','sequence_heavy', 'sequence_light'] ]

    data_list = []

    for index, (study, name, label_stat, h_data, l_data) in enumerate(data.values):
        heavy_prim, heavy_len = h_data, len(h_data)
        light_prim, light_len = l_data, len(l_data)
        label_id = label_stat

        data_list.append({
            #"study" : study,
            #"name" : name,
            "heavy_data": (heavy_prim, heavy_len),
            "light_data": (light_prim, light_len),
            "metadata" : int(label_id)
        })
        
    return data_list


def sequences_to_h5(csv_file, out_file_path, overwrite=False):
    
    data_list = process_csv_data(csv_file)
    
    num_seqs = len(data_list)
    max_h_len = 200
    max_l_len = 200
    
    if overwrite and os.path.isfile(out_file_path):
        os.remove(out_file_path)
    
    h5_out = h5py.File(out_file_path, 'w')
    h_len_set = h5_out.create_dataset('heavy_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)

    l_len_set = h5_out.create_dataset('light_chain_seq_len', (num_seqs, ),
                                      compression='lzf',
                                      dtype='uint16',
                                      maxshape=(None, ),
                                      fillvalue=0)

    h_prim_set = h5_out.create_dataset('heavy_chain_primary',
                                       (num_seqs, max_h_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_h_len),
                                       fillvalue=-1)

    l_prim_set = h5_out.create_dataset('light_chain_primary',
                                       (num_seqs, max_l_len),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, max_l_len),
                                       fillvalue=-1)

    label_set = h5_out.create_dataset('label', (num_seqs, ),
                                        compression='lzf',
                                        dtype="i")
                                        #dtype='uint16',
                                        #fillvalue=0)

    index_set = h5_out.create_dataset('index', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())


    for index, data_dict in enumerate(data_list):
        heavy_prim, heavy_len = data_dict["heavy_data"]
        light_prim, light_len = data_dict["light_data"]
        metadata = data_dict["metadata"]

        h_len_set[index] = heavy_len
        l_len_set[index] = light_len

        h_prim_set[index, :len(heavy_prim)] = np.array(
                letter_to_num(heavy_prim, _aa_dict))
        l_prim_set[index, :len(light_prim)] = np.array(
                letter_to_num(light_prim, _aa_dict))
        label_set[index] = metadata


def cli():
    # Define project path?
    # Define path to data directory?

    desc = 'Creates h5 files from the FASTA sequences of a TS50 study'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--csv_filename', type=str,
        help='The name of the CSV file with the sequences'
    )
    parser.add_argument('--out_file', type=str, default='align-seq.h5',
        help='The name of the output h5 file.'
    )
    parser.add_argument('--overwrite', type=bool, default=True,
        help='whether to overwrite the file or not'
    )

    args=parser.parse_args()
    csv_filename = args.csv_filename
    out_file = args.out_file
    overwrite = args.overwrite

    sequences_to_h5( csv_filename, out_file, overwrite=overwrite )


if __name__ == '__main__':
    cli()


