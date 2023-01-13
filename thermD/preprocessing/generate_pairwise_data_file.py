import h5py
import numpy as np
import argparse
from tqdm import tqdm
import os
from os import listdir, remove
from os.path import join
from os.path import basename, splitext
from glob import glob
import pandas as pd
import json
import itertools
import thermD
from thermD.utility import utils as utils
from thermD.utility.utils import _aa_dict, letter_to_num, protein_pairwise_energy_data

def process_seq_data( csv_file ):
    
    col_names = pd.read_csv(csv_file, nrows=0).columns
    data = pd.read_csv(csv_file, skiprows=1, names=col_names, header=None)
    data = data[ ['Study','Name','Label','sequence_heavy', 'sequence_light'] ]

    data_list = {}

    for index, (study, name, label_stat, h_data, l_data) in enumerate(data.values):
        heavy_prim, heavy_len = h_data, len(h_data)
        light_prim, light_len = l_data, len(l_data)
        label_id = label_stat
        
        name = name.replace("(","").replace(")","").replace("'","").replace("\\","")

        if study not in data_list:
            data_list[study] = {}
        
        data_list[study][name] = {
            "heavy_data": (heavy_prim, heavy_len),
            "light_data": (light_prim, light_len),
            "metadata" : int(label_id)
        }
        
    return data_list


def process_all_data( csv_file ):
    
    col_names = pd.read_csv(csv_file, nrows=0).columns
    data = pd.read_csv(csv_file, skiprows=1, names=col_names, header=None)
    #data = data[ ['Study','Name','Label','sequence_heavy', 'sequence_light',
    #              'ddG','ddG-Interface','SASA-polar','SASA-hphobic','nres-int'] ]
    data = data[ ['Study','Name','Label','TS50', 'sequence_heavy', 'sequence_light',
                  'ddG','ddG-Interface','SASA-polar','SASA-hphobic','nres-int'] ]
    data_list = {}

    for index, (study, name, label_stat, ts50_value, h_data, 
                l_data, ddg, ddg_int, sasa_p, sasa_h, nres_int) in enumerate(data.values):
        heavy_prim, heavy_len = h_data, len(h_data)
        light_prim, light_len = l_data, len(l_data)
        label_id = label_stat
        ddg_val = ddg
        dIsc_val = ddg_int
        psasa_val = sasa_p
        hsasa_val = sasa_h
        nres = nres_int

        name = name.replace("(","").replace(")","").replace("'","").replace("\\","")

        if study not in data_list:
            data_list[study] = {}

        data_list[study][name] = {
            #"token" : study+name,
            "token" : ts50_value,
            "heavy_data": (heavy_prim, heavy_len),
            "light_data": (light_prim, light_len),
            "metadata" : int(label_id),
            "ddG" : (ddg_val, 1),
            "ddG_int" : (dIsc_val, 1),
            "sasa_p" : (psasa_val, 1),
            "sasa_h" : (hsasa_val, 1),
            "nres_int" : (nres, 1)
        }
        
    return data_list


def pairwise_to_h5( energy_dir, csv_file, out_file_path, overwrite=False, print_progress=False ):
    """Convert the data to a hdf5 file
    """
    energy_files = [_ for _ in listdir(energy_dir) if _[-3:] == 'out']
    num_seqs = len(energy_files)

    data_list = process_all_data(csv_file)

    max_h_len = 250
    max_l_len = 250
    max_total_len = 500

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
    
    ddg_set = h5_out.create_dataset('ddG', (num_seqs, 1),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, 1),
                                       fillvalue=-1)

    ddg_int_set = h5_out.create_dataset('ddG_int', (num_seqs, 1),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, 1),
                                       fillvalue=-1)

    sasa_polar_set = h5_out.create_dataset('sasa_polar', (num_seqs, 1),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, 1),
                                       fillvalue=-1)

    sasa_hphobic_set = h5_out.create_dataset('sasa_hphobic', (num_seqs, 1),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, 1),
                                       fillvalue=-1)

    nres_int_set = h5_out.create_dataset('nres_int', (num_seqs, 1),
                                       compression='lzf',
                                       dtype='uint8',
                                       maxshape=(None, 1),
                                       fillvalue=-1)

    index_set = h5_out.create_dataset('index', (num_seqs, ),
                                        compression='lzf',
                                        dtype=h5py.string_dtype())

    dt = h5py.special_dtype(vlen=bytes)
    #token_set = h5_out.create_dataset('token', (num_seqs,), 
    #                                  dtype=dt)
    token_set = h5_out.create_dataset('token', (num_seqs,), 
                                        compression='lzf',
                                        dtype='float')

    pairwise_energy_set = h5_out.create_dataset('pairwise_energy_mat', 
                                                (num_seqs, max_total_len, max_total_len),
                                                maxshape=(None, max_total_len, max_total_len),
                                                compression='lzf',
                                                dtype='float',
                                                fillvalue=99)

    
    for index, input_file in tqdm( enumerate(energy_files), disable=(not print_progress),
                            total=len(energy_files)):
        # Obtain the names of all the files

        study_id, name_id = utils.get_energy_id(input_file)
        if name_id not in data_list[study_id]:
            print("Warning: Not found {} in the database of {}".format(name_id, study_id))
            continue
        else:
            info = data_list[ study_id ][ name_id ]
            i_file = str( os.path.join( energy_dir, input_file ) )
            pairwise_energy_mat = protein_pairwise_energy_data( i_file )
            info.update( dict(pairwise_energy=pairwise_energy_mat) )

            heavy_prim, heavy_len = info["heavy_data"]
            light_prim, light_len = info["light_data"]
            metadata = info["metadata"]
            token = info["token"]
            ddG, ddG_len = info["ddG"]
            ddG_int, ddG_int_len = info["ddG_int"]
            sasa_p, sasa_p_len = info["sasa_p"]
            sasa_h, sasa_h_len = info["sasa_h"]
            nres_int, nres_int_len = info["nres_int"]

            total_len = len(heavy_prim) + len(light_prim)

            try:
                pairwise_energy_set[
                    index, :total_len, :total_len] = np.array(
                        info['pairwise_energy']
                    )
            except TypeError:
                msg = ('Fasta/PDB coordinate length mismatch: the fasta sequence '
                    'length of {} and the number of coordinates ({}) in {} '
                    'mismatch.\n ')
                raise ValueError(
                    msg.format(total_len, len(info['pairwise_energy']), input_file))
            
            h_len_set[index] = heavy_len
            l_len_set[index] = light_len

            h_prim_set[index, :len(heavy_prim)] = np.array(
                    letter_to_num(heavy_prim, _aa_dict))
            l_prim_set[index, :len(light_prim)] = np.array(
                    letter_to_num(light_prim, _aa_dict))
            label_set[index] = metadata
            token_set[index] = token
            ddg_set[index, :1] = np.array( ddG )
            ddg_int_set[index, :1] = np.array( ddG_int )
            sasa_polar_set[index, :1] = np.array( sasa_p )
            sasa_hphobic_set[index, :1] = np.array( sasa_h )
            nres_int_set[index, :1] = np.array( nres_int )


def cli():

    desc = 'Creates h5 files from the datafile (sequences + energies) of a TS50 study'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        '--energy_dir',
        type=str,
        help='The directory containing score files where an antibody'
        'with the score file of the PDB of a particular sequence is named: ID.out')
    
    parser.add_argument('--csv_filename', type=str,
        help='The name of the CSV file with the sequences'
    )
    parser.add_argument('--out_file', type=str, default='data-seq.h5',
        help='The name of the output h5 file.'
    )
    parser.add_argument('--overwrite', type=bool, default=True,
        help='whether to overwrite the file or not'
    )

    args=parser.parse_args()
    energy_dir = args.energy_dir
    csv_filename = args.csv_filename
    out_file = args.out_file
    overwrite = args.overwrite


    pairwise_to_h5( energy_dir, csv_filename, out_file, overwrite=overwrite )


if __name__ == '__main__':
    cli()



    





