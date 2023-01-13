#!usr/bin/env python

'''
Scoring script for Total energy
Author: Ameya Harmalkar
Date: 14th June 2021
'''

import optparse
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

def parse_score_file(filename, score_term):
    '''
    Takes in the scorefile and extracts the difference with respect
    to the wildtype for the stated score-term
    
    **Parameters**
        
        filename : *str*
        The name of the scorefile with the mutant data
        score_term : *str*
        The score term that needs to be compared
    '''
    df = pd.read_csv(filename, delim_whitespace = True, header=0, skiprows = [0])
    
    scores = [ float(x) for x in df[score_term]  ]
    description = [ x for x in df['description'] ]
    
    energy_dict = {}
    
    for i in range( len(description) ):
        index_val = description[i].split('_msa_')
        key = index_val[0] + '_' + index_val[1].split('_complex_')[0]
        if( key not in energy_dict ):
            energy_dict[key] = []
            energy_dict[key].append(float(scores[i]))
        elif key in energy_dict:
            energy_dict[key].append(float(scores[i]))
        else:
            sys.exit("DANGER WILL ROBINSON!")
    
    final_dict = {}
    
    for k,v in energy_dict.items():
        value = np.mean( energy_dict[k] )
        final_dict[k] = value
        
    return final_dict



def regularize( number ):
    reg_value = map( lambda x: float( ( x- np.min(number) ) /  ( np.max(number) - np.min(number) ) ), number )
    return list(reg_value)


def generate_binned_csv(filename):
    
    output = filename.strip('_score.sc')
    labels = list(range(0,20))
    
    energy_dict = parse_score_file(filename, 'total_score')
    interface_dict = parse_score_file(filename, 'IA_dG_separated')
    sasa_P = parse_score_file(filename, 'IA_dSASA_polar')
    sasa_H = parse_score_file(filename, 'IA_dSASA_hphobic')
    nres_int = parse_score_file(filename, 'IA_nres_int')
    
    data_list = []
    
    for k,v in energy_dict.items():
        key = k.lstrip(output).lstrip('_')
        score_terms = [ str(output), str(key),float(energy_dict[k]), float(interface_dict[k]), 
                       float(sasa_P[k]), float(sasa_H[k]), float(nres_int[k]) ]
        data_list.append(score_terms)
        
    data_list = list(map(list, zip(*data_list)))
    
    data_list[2] = pd.cut( regularize(data_list[2]), 20, labels = labels)
    data_list[3] = pd.cut( regularize(data_list[3]), 20, labels = labels)
    data_list[4] = pd.cut( regularize(data_list[4]), 20, labels = labels)
    data_list[5] = pd.cut( regularize(data_list[5]), 20, labels = labels)
    data_list[6] = pd.cut( regularize(data_list[6]), 20, labels = labels)
    
    #data_out = np.histogram(data_list[1], 20)[0]
    
    dataset = pd.DataFrame(np.transpose(data_list))
    dataset.columns = [ 'Study',  'Name', 'ddG', 'ddG-Interface', 'SASA-polar', 'SASA-hphobic', 'nres-int' ]
    dataset.to_csv(output+'_bin.csv', index=False)
    
    return data_list


def generate_csv_file(filename):
    
    output = filename.strip('_score.sc')
    labels = list(range(0,20))
    
    energy_dict = parse_score_file(filename, 'total_score')
    interface_dict = parse_score_file(filename, 'IA_dG_separated')
    sasa_P = parse_score_file(filename, 'IA_dSASA_polar')
    sasa_H = parse_score_file(filename, 'IA_dSASA_hphobic')
    nres_int = parse_score_file(filename, 'IA_nres_int')
    
    data_list = []
    
    for k,v in energy_dict.items():
        key = k.lstrip(output).lstrip('_')
        score_terms = [ str(output), str(key), float(energy_dict[k]), float(interface_dict[k]), 
                       float(sasa_P[k]), float(sasa_H[k]), float(nres_int[k]) ]
        data_list.append(score_terms)
        
    data_list = list(map(list, zip(*data_list)))
    
    data_list[2] = regularize(data_list[2])
    data_list[3] = regularize(data_list[3])
    data_list[4] = regularize(data_list[4])
    data_list[5] = regularize(data_list[5])
    data_list[6] = regularize(data_list[6])

    
    dataset = pd.DataFrame(np.transpose(data_list))
    dataset.columns = [ 'Study', 'Name', 'ddG', 'ddG-Interface', 'SASA-polar', 'SASA-hphobic', 'nres-int' ]
    dataset.to_csv(output+'.csv', index=False)
    
    return dataset

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '-s', '--src', help='Name of FASTA to extract sequences from', required=True )
    #parser.add_argument( '-d', '--des', help='Name of outputfile', required=True )
    args = parser.parse_args()
    src = args.src

    generate_csv_file(src)
    print('Done generating datafile')
    generate_binned_csv(src)
    print('Binning complete!')

    


if __name__ == "__main__":
    main()