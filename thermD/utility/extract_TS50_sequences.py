#!usr/bin/env python

"""Script to obtain a csv file from MSA data
"""

import argparse
import os
import pandas as pd
import numpy
import shutil
import math


def check_TS_label(value):

    if( value == 'up' ):
        return 3
    elif( value == 'neg' or value == 'nan' or value == 'no'):
        return 0
    elif( float(value) < 50):
        return 0
    #elif( 40 <= float(value) < 50):
    #    return 1
    elif( 50 <= float(value) < 60):
        return 1
    elif( 60 <= float(value) < 70):
        return 2
    elif( float(value) >= 70):
        return 3
    else:
        print('Error: TS50 labels have unidentified label. ' + str(value) )
        return 0


def check_experiment_label(value):

    study_dict = {
        'CCR8' : 0,
        'CD123' : 1,
        'CD20' : 2,
        'CD22AFFMAT' : 3,
        'CD22starting' : 4,
        'CD70' : 5,
        'CLDN18.2' : 6,
        'CLL1' : 7,
        'CLL1opt' : 7,
        'CS1' : 8,
        'CS115G8Opt' : 9,
        'CS16C12Opt' : 9,
        'DCAF4L2' : 10,
        'EpCAM' : 11,
        'LRRC15' : 12,
        'MAGEB2' : 13,
        'MUC13' : 14,
        'MUC16' : 15,
        'MUC17' : 16,
        'MUC1-SEA' : 17,
        'PSMAAFFMAT' : 18
    }

    return study_dict[value]



def split_fasta(filename, outfilename):

    file = open(filename, 'r')
    outfile = open(outfilename+'.csv', 'w')

    #outfile.write('Study,Name,sequence_heavy,sequence_light,length\n')

    key = filename.split('/')[-1].split('_')[0]
    for line in file.readlines():
        if line.startswith('>'):
            outfile.write(key + ',')
            line = line.split('|')
            out_name = line[0].lstrip('>').rstrip()
            ts_label = check_TS_label(line[1].strip())
            #st_label = check_experiment_label(key)
            outfile.write( out_name + ',' + str(ts_label) + ',')
        else:
            data = line.split('GGGGS')
            outfile.write( data[0].strip() + ',' + data[-1].strip()  +  "\n" )
    
    outfile.close()
    file.close()

    return 0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '-s', '--src', help='Name of FASTA to extract sequences from', required=True )
    parser.add_argument( '-d', '--des', help='Name of outputfile', required=True )
    args = parser.parse_args()
    src = args.src
    des = args.des

    split_fasta(src, des)

    print('Done generating datafile')


if __name__ == "__main__":
    main()



        