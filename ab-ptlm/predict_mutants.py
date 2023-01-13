from typing import List
import argparse
from pathlib import Path
import subprocess
import pickle as pkl
import pandas as pd
from evo.parsing import read_sequences
from evo.sequence import create_mutant_df
from inference import SupervisedAntiBERTyScorer


parser = argparse.ArgumentParser()
parser.add_argument("fasta", type=Path, help="Fasta file with the sequence to run predictions on")
parser.add_argument(
    "--model", 
    choices=["antiberty_supervised"], # Need to add the unsupervised model here
    default="antiberty_supervised",
    required=True
)
parser.add_argument(
    "--pos",
    type=int,
    default=120,
    help="Residue position indicating the split between heavy and light chains",
    required=True,
)
args = parser.parse_args()

linkers_list = [
            "GGGGS" * 3,  # G4Sx3
            "GGGGS" * 2,  # G4Sx2
            "GGGGSGGGSGGGGS",  # G4SG3SG4S - likely mistranscription
            "GGGGSGGGGPGGGGS",  # G4SG4PG4S - likely mistranscription
        ]

def split_linkers(seq):
    for linker in linkers_list:
        try:
            heavy, light = seq.split(linker)
        except ValueError:
            continue
            
        return heavy, light
    else:
        raise RuntimeError(f"No appropriate linker found: {seq}")


headers, sequences = read_sequences(args.fasta)


def run_antiberty_single_mutant_scan():

    model = SupervisedAntiBERTyScorer()
    data = {}

    pos = 120

    data_list = []
    
    for header, seq in zip(headers, sequences):
        df = create_mutant_df(seq)
        seq_list = []
        for mut, seq in zip( df.mutant, df.sequence ):
            heavy, light = seq[:pos], seq[pos:]
            seq_list.append( {'H': heavy, 'L': light} )
        
        #df["score"] = model.score(df["sequence"])
        output = model.score(seq_list)
        df_sample = pd.DataFrame(output, columns = ["TS50","bin1","bin2","bin3","bin4","predictions"] )
        df_concat = pd.concat( [df, df_sample], ignore_index=False, axis=1 )
        data[header] = df_concat
    
    df_concat.to_csv( './single-mutants/antiBERTy_preds.csv', index = False)
    
    with open("./single-mutants/antiberty_supervised_mutants.pkl", "wb") as f:
        pkl.dump(data, f)


if args.model == "antiberty_supervised":
    run_antiberty_single_mutant_scan()
else:
    raise ValueError(args.model)


