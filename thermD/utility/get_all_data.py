#!usr/bin/env python

'''
Scoring script for Total energy
Author: Ameya Harmalkar
Date: 28th July 2021
'''

import argparse
import os
import statistics
import pandas as pd
import numpy as np
import shutil
import collections
import math
import string
from string import ascii_lowercase 
from string import ascii_uppercase


def get_data_csv( fastafile, binfile, study ):
    
    fasta = pd.read_csv(fastafile, header = None)
    fasta_list = fasta.values.tolist()
    binout = pd.read_csv(binfile, header=0)
    binout_list = binout.values.tolist()
    outfilename = study + '-all.csv'

    fasta_dict = {}
    binout_dict = {}

    if ( len(fasta_dict) != len(binout_dict) ):
        print("Note: Length mismatch in the files!")

    for i in range(len(fasta_list)):
        key = fasta_list[i][1]
        if key not in fasta_dict:
            fasta_dict[key] = {}
            fasta_dict[key]['TS50'] = fasta_list[i][2]
            fasta_dict[key]['VH'] = fasta_list[i][3]
            fasta_dict[key]['VL']  = fasta_list[i][3]
        else:
            print("Danger! Will Robinson!")


    binout_dict = {}
    for i in range(len(binout_list)):
        key = binout_list[i][1]
        if key not in binout_dict:
            binout_dict[key] = {}
            binout_dict[key]['ddG'] = binout_list[i][2]
            binout_dict[key]['ddG-Interface'] = binout_list[i][3]
            binout_dict[key]['SASA-polar'] = binout_list[i][4]
            binout_dict[key]['SASA-hphobic'] = binout_list[i][5]
            binout_dict[key]['Nres-int'] = binout_list[i][6]
        else:
            print("Danger! Will Robinson!")
            
    data_list = []
    for k,v in fasta_dict.items():
        k_strip = k.replace("(","").replace(")","")
        if k in binout_dict:
            score_terms = [ study, str(k), fasta_dict[k]['TS50'], fasta_dict[k]['VH'], fasta_dict[k]['VL'],
                           binout_dict[k]['ddG'], binout_dict[k]['ddG-Interface'], binout_dict[k]['SASA-polar'], binout_dict[k]['SASA-hphobic'], binout_dict[k]['Nres-int'] ]
            data_list.append(score_terms)
        elif k_strip in binout_dict:
            score_terms = [ study, str(k), fasta_dict[k]['TS50'], fasta_dict[k]['VH'], fasta_dict[k]['VL'],
                           binout_dict[k_strip]['ddG'], binout_dict[k_strip]['ddG-Interface'], binout_dict[k_strip]['SASA-polar'], binout_dict[k_strip]['SASA-hphobic'], binout_dict[k_strip]['Nres-int'] ]
            data_list.append(score_terms)
        else:
            print('Error for key : ', k)

    data_list = list(map(list, zip(*data_list)))

    dataset = pd.DataFrame(np.transpose(data_list))
    #dataset.columns = [ 'Study', 'Name', 'ddG', 'ddG-Interface', 'SASA-polar', 'SASA-hphobic', 'nres-int' ]
    dataset.to_csv(outfilename, index=False, header=False)
    
    return 0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '-s', '--src', help='Name of FASTA to extract sequences from', required=True )
    parser.add_argument( '-b', '--bnf', help='Name of binned CSV file', required=True )
    parser.add_argument( '-t', '--std', help='Name of experiment study', required=True )
    args = parser.parse_args()
    src = args.src
    bnf = args.bnf
    std = args.std

    get_data_csv( src, bnf, std)
    print("Generated complete datafile!!")

    


if __name__ == "__main__":
    main()