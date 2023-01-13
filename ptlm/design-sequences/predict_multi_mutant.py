from typing import Sequence
import argparse
import pickle as pkl
import pandas as pd
from itertools import combinations, islice
import torch
import numpy as np
from pathlib import Path
from evo.sequence import make_mutation, mutant_to_names
from data_loading import load_sequences_and_cdr, FASTQ_FILE
from scoring import UnsupervisedESM1vScorer, SupervisedESM1bScorer

SINGLE_MUTANT_DATA = Path(__file__).with_name("single-mutants")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", choices=["esm1v_unsupervised", "esm1b_supervised"], required=True
)
parser.add_argument(
    "--fastq", type=Path, 
    help="Path to fastq file with sequences to download. "
         "Second line should contain spaces and '*', with a '*' "
         "in every position that should not be mutated.",
    default=FASTQ_FILE,
)
parser.add_argument(
    "--max_mutations", type=int, default=3,
    help="Maximum number of allowable mutations"
)
parser.add_argument(
    "--single_mutant_data_file", type=Path, default=None,
    help="path to output of `predict_single_mutant.py`"
)
parser.add_argument(
    "--num_mutants_to_score", type=int, default=120,
    help="total number of mutants to generate and score"
)
args = parser.parse_args()

# 1. Read in single mutant data (make sure to run this first!)
if args.single_mutant_data_file is None:
    args.single_mutant_data_file = SINGLE_MUTANT_DATA / f"{args.model}_mutants.pkl"
if not args.single_mutant_data_file.exists():
    raise RuntimeError("Please run `predict_single_mutant.py` first and store the result in single-mutants")
with open(args.single_mutant_data_file, "rb") as f:
    data = {name.removesuffix(".csv"): df for name, df in pkl.load(f).items()}

headers, sequences, regions = load_sequences_and_cdr()
seqs = dict(zip(headers, sequences))
regions = {
    header: np.where(np.array(list(region)) != "*")[0] + 1
    for header, region in zip(headers, regions)
}

# 2. Create Top Mutants

def is_valid_combination(mutants: Sequence[str]) -> bool:
    positions = {mutant_to_names(mut)[1] for mut in mutants}
    return len(positions) == len(mutants)

mutants = {}
for name, df in data.items():
    df["position"] = df["mutant"].apply(lambda x: mutant_to_names(x)[1] if x != "WT" else -1)
    position_is_valid = df["position"].isin(regions[name])
    mutant_is_valid = ~df["mutant"].str.contains("C")  # no mutants involving cysteines
    df = df.loc[position_is_valid & mutant_is_valid]
    df = df.sort_values("score", ascending=False)
    best_mutants = df[["mutant", "score"]]
    mut_names = [f"mut{i}" for i in range(args.max_mutations)]
    all_mutants = pd.DataFrame(
        list(islice(filter(is_valid_combination, combinations(best_mutants["mutant"], args.max_mutations)), args.num_mutants_to_score)),
        columns=mut_names,
    )
    all_mutants["sum_score"] = sum(
        best_mutants.set_index("mutant")
        .loc[all_mutants[name]]
        .reset_index(drop=True)
        for name in mut_names
    )
    all_mutants["mutant"] = all_mutants.apply(
        lambda row: ",".join(row[mut_names]), axis=1
    )
    mutants[name] = all_mutants

# 3. Re-score Mutants
torch.set_grad_enabled(False)

if args.model == "esm1v_unsupervised":
    model = UnsupervisedESM1vScorer()
elif args.model == "esm1b_supervised":
    model = SupervisedESM1bScorer()
else:
    raise ValueError(args.model)

for name, df in mutants.items():
    for mutant in df["mutant"]:
        make_mutation(seqs[name], mutant)
    sequences = [make_mutation(seqs[name], mutant) for mutant in df["mutant"]]
    scores = model.score(sequences)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    df["rescore"] = scores

with open(f"./multi-mutants/{args.model}_multi_mutants.pkl", "wb") as f:
    pkl.dump(mutants, f)
