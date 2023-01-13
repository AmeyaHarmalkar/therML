from itertools import product
from pathlib import Path
import pickle as pkl
import numpy as np
from mcmc import SupervisedESM1bMCMC, UnsupervisedESM1vMCMC
from data_loading import load_sequences_and_cdr, FASTQ_FILE
import argparse

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
    "--num_mcmc_steps", type=int, default=10000,
    help="Number of steps of MCMC to run in each trajectory"
)
parser.add_argument(
    "--num_restarts", type=int, default=3,
    help="Number of trajectories to run for each sequence"
)
args = parser.parse_args()

headers, sequences, regions = load_sequences_and_cdr()

trajectories = {}
try:
    for (header, sequence, region), restart in product(
        zip(headers, sequences, regions), range(args.num_restarts)
    ):
        print(f"Running {header}, restart {restart + 1} / {args.num_restarts}")
        valid_indices = np.where(np.asarray(list(region)) != "*")[0] + 1
        if args.model == "esm1v_unsupervised":
            mcmc = UnsupervisedESM1vMCMC(
                args.num_mcmc_steps, args.max_mutations, mask_positions=False, no_cysteine=True
            )
        elif args.model == "esm1b_supervised":
            mcmc = SupervisedESM1bMCMC(args.num_mcmc_steps, args.max_mutations, no_cysteine=True)
        else:
            raise ValueError(args.model)

        trajectory = mcmc(sequence, valid_indices)
        trajectories[(header, restart)] = trajectory
finally:
    with open(f"./multi-mutants/{args.model}_mcmc_outputs.pkl", "wb") as f:
        pkl.dump(trajectories, f)
