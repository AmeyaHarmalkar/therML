#!usr/bin/env python

"""Script to obtain a csv file from MSA data
"""

import argparse
import os
from os import listdir
from os.path import basename, splitext
import pandas as pd
import numpy
import shutil
import math


def get_chains(filename):
    
    file = open(filename, 'r')
    key = splitext(basename(filename))[0]
    data_dict = {}
    data_dict['Name'] = key
    for line in file.readlines():
        if line.startswith('>'):
            idx = line = line.split('|')[-1].rstrip()
        else:
            data_dict[idx] = line.rstrip()
    
    return data_dict


def split_fasta_dir(fasta_dir, outfilename, study):
    
    fasta_files = [ os.path.join(fasta_dir,_) for _ in listdir(fasta_dir) if _[-5:] == 'fasta' ]
    num_files = len(fasta_files)
    
    outfile = open(outfilename+'.csv', 'w')

    outfile.write('Study,Name,Label,sequence_heavy,sequence_light,length\n')
    
    for i in range(num_files):
        data_dict = get_chains(fasta_files[i])
        outfile.write( study + ',' + data_dict['Name'] + ',99,' +data_dict['H'] + ',' + data_dict['L'] + "\n")
    
    outfile.close()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument( '-s', '--src', help='Name of FASTA directory with all the FASTA files', required=True )
    parser.add_argument( '-d', '--des', help='Name of outputfile', required=True )
    parser.add_argument( '-t', '--std', help='Name of the experiment', required=True)
    args = parser.parse_args()
    src = args.src
    des = args.des
    std = args.std

    split_fasta_dir(src, des, std)

    print('Done generating datafile')


if __name__ == "__main__":
    main()
