from typing import List
import argparse
from pathlib import Path
import subprocess
import pickle as pkl
import pandas as pd
from evo.parsing import read_sequences
from evo.sequence import create_mutant_df
import tempfile
from scoring import SupervisedESM1bScorer
from data_loading import load_sequences_and_cdr

parser = argparse.ArgumentParser()
parser.add_argument("fasta", type=Path, help="Fasta file with sequences to run prediction on.")
parser.add_argument(
    "--model", choices=["esm1v_unsupervised", "esm1b_supervised"], required=True
)
parser.add_argument(
    "--scoring-strategy",
    choices=["wt-marginals", "masked-marginals"],
    default="masked-marginals",
    help="If using esm-1v, which scoring strategy to use."
)
args = parser.parse_args()

if args.fasta.suffix == ".fastq":
    headers, sequences, _ = load_sequences_and_cdr(args.fasta)
else:
    headers, sequences = read_sequences(args.fasta)

def run_esm1v_single_mutant_scan():

    def make_command(header: str, sequence: str, filename: str) -> List[str]:
        command = (
            " ".join(
                [
                    "python esm1v_single_mutant_scan.py",
                    f"--sequence {sequence}",
                    f"--dms-input {filename}",
                    "--offset-idx 1",
                    f"--scoring-strategy {args.scoring_strategy}",
                    "--model-location",
                ]
            )
            + " "
            + " ".join(f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6))
        )
        command = command.split()
        command.append("--dms-output")
        command.append(f"./single-mutants/esm1v_unsupervised_{header}.csv")
        print(" ".join(command))
        return command

    for header, seq in zip(headers, sequences):
        df = create_mutant_df(seq)
        with tempfile.NamedTemporaryFile() as f:
            df.to_csv(f.name, index=False)
            subprocess.run(make_command(header, seq, f.name))
    for header in headers:
        data = {header: pd.read_csv(f"./single-mutants/esm1v_unsupervised_{header}.csv")}
        for df in data.values():
            df["score"] = df[[f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]].mean(1)
        with open("./single-mutants/esm1v_unsupervised_mutants_new.pkl", "wb") as f:
            pkl.dump(data, f)

def run_esm1b_single_mutant_scan():
    model = SupervisedESM1bScorer()
    data = {}
    for header, seq in zip(headers, sequences):
        df = create_mutant_df(seq)
        df["score"] = model.score(df["sequence"])
        data[header] = df
    with open("./single-mutants/esm1b_supervised_mutants.pkl", "wb") as f:
        pkl.dump(data, f)


if args.model == "esm1v_unsupervised":
    run_esm1v_single_mutant_scan()
elif args.model == "esm1b_supervised":
    run_esm1b_single_mutant_scan()
else:
    raise ValueError(args.model)
