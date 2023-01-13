from typing import Tuple, List
from pathlib import Path
import json
import numpy as np
from evo.parsing import read_sequences


DATA_DIR = Path(__file__).with_name("data")
FASTA_FILE = DATA_DIR / "scFvs_to_design.fasta"
CDR_FILE = DATA_DIR / "cdr_region.json"
FASTQ_FILE = FASTA_FILE.with_suffix(".fastq")


def load_sequences(path=FASTA_FILE) -> Tuple[List[str], List[str]]:
    return read_sequences(path)


def load_sequences_and_cdr(path=FASTQ_FILE) -> Tuple[List[str], List[str], List[str]]:
    path = Path(path)
    headers, sequences, regions = [], [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 3 == 0:
                headers.append(line[1:].strip("\n"))
            elif i % 3 == 1:
                sequences.append(line.strip("\n"))
            elif i % 3 == 2:
                regions.append(line.strip("\n"))
    return headers, sequences, regions


def mark_cdr_regions():
    ordering = ["2B12", "2B12", "CD40", "CD19", "CD19", "MSLN", "MSLN"]

    with open(DATA_DIR / "cdr_region.json") as f:
        cdr_regions = json.load(f)

    headers, sequences = load_sequences()

    with open(FASTQ_FILE, "w") as f:
        for header, sequence, name in zip(headers, sequences, ordering):
            mark_valid = np.array([" " for _ in sequence])
            for region in cdr_regions[name]:
                index = sequence.index(region)
                mark_valid[index:index + len(region)] = "*"

            # also mark linker as invalid
            index = sequence.index("GGGGS" * 3)
            mark_valid[index:index + 15] = "*"
            f.write(f">{header}\n{sequence}\n{''.join(mark_valid)}\n")
