#!usr/bin/env python

"""Script to obtain a csv file from MSA data
"""

import argparse
import os
import pandas as pd
import numpy
import shutil
import math


def split_fasta(filename, outfilename):

    file = open(filename, 'r')
    outfile = open(outfilename+'.csv', 'w')

    outfile.write('Study,Name,sequence_heavy,sequence_light,length\n')

    key = filename.split('_')[0].split('/')[-1]
    for line in file.readlines():
        if line.startswith('>'):
            outfile.write(key + ',')
            out_name = line.lstrip('>').rstrip()
            outfile.write( out_name + ',' )
        else:
            length = len(line)
            line = line.split(';')
            outfile.write( line[0].strip() + ',' + line[1].strip()  + ',' + str(length) + "\n" )
    
    outfile.close()

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



        