import optparse, argparse
import os
import statistics
import pandas as pd
import numpy as np
import shutil
import collections
import math
import seaborn as sns
import chart_studio.plotly as py
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

def parse_score_file(filename, score_term, nativefile):
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
    df_native = pd.read_csv(nativefile, delim_whitespace = True, header=0, skiprows = [0])
    
    native_score = np.mean( [ float(x) for x in df_native[score_term]  ] )
    print( "Native score: ", native_score)
    scores = [ float(x) for x in df[score_term]  ]
    description = [ x for x in df['description'] ]
    
    energy_dict = {}
    
    for i in range( len(description) ):
        index_val = description[i].split('_0001')
        key = index_val[0]
        if( key not in energy_dict ):
            energy_dict[key] = []
            energy_dict[key].append(float(scores[i] - native_score ))
        elif key in energy_dict:
            energy_dict[key].append(float(scores[i] - native_score ))
        else:
            sys.exit("DANGER WILL ROBINSON!")
    
    final_dict = {}
    
    for k,v in energy_dict.items():
        value = np.min( energy_dict[k] )
        final_dict[k] = value
        
    return final_dict


def generate_csv_file(filename, nativefile, outfile, study):
    
    output = outfile
    
    energy_dict = parse_score_file(filename, 'total_score', nativefile)
    interface_dict = parse_score_file(filename, 'IA_dG_separated', nativefile)
    sasa_P = parse_score_file(filename, 'IA_dSASA_polar', nativefile)
    sasa_H = parse_score_file(filename, 'IA_dSASA_hphobic', nativefile)
    nres_int = parse_score_file(filename, 'IA_nres_int', nativefile)
    
    data_list = []
    
    for k,v in energy_dict.items():
        #key = k.lstrip(output).lstrip('_')
        key = k
        score_terms = [ str(study), str(key), float(energy_dict[k]), float(interface_dict[k]), 
                       float(sasa_P[k]), float(sasa_H[k]), float(nres_int[k]) ]
        data_list.append(score_terms)
        
    data_list = list(map(list, zip(*data_list)))
    
    data_list[2] = data_list[2]
    data_list[3] = data_list[3]
    data_list[4] = data_list[4]
    data_list[5] = data_list[5]
    data_list[6] = data_list[6]

    
    dataset = pd.DataFrame(np.transpose(data_list))
    dataset.columns = [ 'Study', 'Name', 'ddG', 'ddG-Interface', 'SASA-polar', 'SASA-hphobic', 'nres-int' ]
    dataset.to_csv( output +'.csv', index=False)
    
    return dataset


def aggregate_data(energy_dataset, prediction_csv, output):
    
    df_energy = energy_dataset

    ddg = [x for x in df_energy['ddG']]
    dIsc = [x for x in df_energy['ddG-Interface']]
    SASA_p = [x for x in df_energy['SASA-polar']]
    SASA_h = [x for x in df_energy['SASA-hphobic']]
    description = [x for x in df_energy['Name']]

    energy_labels = {}

    for i in range(len(description)):
        if description[i] not in energy_labels:
            energy_labels[description[i]] = []
        energy_labels[description[i]].append(ddg[i])
        energy_labels[description[i]].append(dIsc[i])
        energy_labels[description[i]].append(float(SASA_p[i])+float(SASA_h[i]))
    
    df_pred = pd.read_csv(prediction_csv, header = 0) 
    idx = [x for x in df_pred['Name']]
    labels = [x for x in df_pred['Predictions']]
    bin1 = [x for x in df_pred['under50-bin']]
    bin2 = [x for x in df_pred['50-60-bin']]
    bin3 = [x for x in df_pred['60-70-bin']]
    bin4 = [x for x in df_pred['70up-bin']]

    
    pred_labels = {}
    for i in range(len(idx)):
        if idx[i] not in pred_labels:
            pred_labels[idx[i]] = []
        pred_labels[idx[i]].append(bin1[i])
        pred_labels[idx[i]].append(bin2[i])
        pred_labels[idx[i]].append(bin3[i])
        pred_labels[idx[i]].append(bin4[i])
        pred_labels[idx[i]].append(labels[i])
        

    print( len(pred_labels), len(energy_labels) )
    
    data_list = []

    for k,v in energy_labels.items():
        if k in pred_labels:
            score_terms = [ k, v[0], v[1], v[2], pred_labels[k][0], 
                           pred_labels[k][1], pred_labels[k][2], pred_labels[k][3], pred_labels[k][4] ]
            data_list.append(score_terms)
        else:
            print(k," missing in the prediction labels")

    data_list = list(map(list, zip(*data_list)))    
    dataset = pd.DataFrame(np.transpose(data_list))
    dataset.columns = [ 'Name', 'ddG', 'ddG-Interface', 'SASA-int', 
                        'under50-bin', '50-60-bin', '60-70-bin', '70up-bin', 'Predictions' ]
    dataset.to_csv( output +'.csv', index=False)
    
    return data_list


def _get_args():
    """Obtain the command line arguments"""
    desc = ('''
        Script for predicting the bins for point mutants with 
        consensus across models trained on different datasets
    ''')

    parser = argparse.ArgumentParser(description=desc)
    
    #training arguments
    parser.add_argument('--srcfile', type=str, 
                        help='name of the native score file')
    parser.add_argument('--natfile', type=str, 
                        help='name of the native score file')
    parser.add_argument('--preds', type=str, 
                        help='name of the prediction file from the network')
    parser.add_argument('--outfile', type=str, default='mutant',
                        help='Name of the output file')
    parser.add_argument('--study', type=str, default='mutant',
                        help='Name of the output file')

    return parser.parse_args()


def _cli():

    args = _get_args()
    srcfile = args.srcfile
    natfile = args.natfile
    preds = args.preds
    outfile = args.outfile
    study = args.study

    dataset = generate_csv_file(srcfile, natfile, outfile, study)

    outfilename = outfile + '_colated'
    aggregate_data(dataset, preds, outfilename)


if __name__ == '__main__':
    _cli()