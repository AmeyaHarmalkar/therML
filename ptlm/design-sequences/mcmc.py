from typing import Tuple, Dict, Sequence, Optional
from pathlib import Path
import sys
from abc import ABC, abstractmethod
import math
import random
from copy import copy
import numpy as np
import pandas as pd
from tqdm import trange
import esm
from evo.sequence import make_mutation, mutant_to_names
from evo.likelihood import sequence_pseudo_ppl
from evo.tokenization import Vocab


_FASTA_VOCAB = np.array(list("ARNDCQEGHILKMFPSTWYV"))
_FASTA_VOCAB_NO_CYS = np.array(list("ARNDQEGHILKMFPSTWYV"))


class MCMC(ABC):
    def __init__(
        self, steps: int = 10000, max_mutants: int = 3, no_cysteine: bool = False
    ):
        self.steps = steps
        self.max_mutants = max_mutants
        self.no_cysteine = no_cysteine
        self.alphabet = _FASTA_VOCAB if not no_cysteine else _FASTA_VOCAB_NO_CYS

    def _add(self, seq: str, likelihood: float, mutants: Dict[int, str]) -> None:
        self.sequences.append(seq)
        self.likelihood.append(likelihood)
        self.mutants.append(",".join(mutants.values()))

    def _accept(self, likelihood: float) -> bool:
        ratio = likelihood / self.likelihood[-1]
        return random.random() < ratio

    def _reverse_mutant(self, mutant: str) -> str:
        wt, pos, mt = mutant_to_names(mutant)
        return f"{mt}{pos}{wt}"

    @abstractmethod
    def compute_likelihood(self, seq: str) -> float:
        raise NotImplementedError

    def _make_move(
        self,
        sequence: str,
        mutants: Dict[int, str],
        valid_indices: Optional[Sequence[int]] = None,
    ) -> Tuple[str, Dict[int, str]]:
        """Modify an input protein in some way, with a maximum number of mutants."""

        mutants = copy(mutants)
        # If we have too many mutants, remove one and call the method on the result
        if len(mutants) == self.max_mutants:
            to_remove = random.choice(list(mutants.keys()))
            reverse = self._reverse_mutant(mutants.pop(to_remove))
            sequence = make_mutation(sequence, reverse)
            return self._make_move(sequence, mutants, valid_indices=valid_indices)

        # Base case, we can make a new mutant
        if valid_indices is None:
            position_to_mutate = random.randint(1, len(sequence))
        else:
            position_to_mutate = random.choice(valid_indices)
        wt = sequence[position_to_mutate - 1]
        mt = random.choice(self.alphabet[self.alphabet != wt])
        sequence = make_mutation(sequence, f"{wt}{position_to_mutate}{mt}")

        if position_to_mutate in mutants:
            existing_mutant = mutants.pop(position_to_mutate)
            wt, _, _ = mutant_to_names(existing_mutant)

        if wt != mt:
            mutant = f"{wt}{position_to_mutate}{mt}"
            mutants[position_to_mutate] = mutant
        return sequence, mutants

    def __call__(
        self, sequence: str, valid_indices: Optional[Sequence[int]] = None
    ) -> pd.DataFrame:

        if self.no_cysteine:
            not_cysteine_indices = np.where(np.array(list(sequence)) != "C")[0] + 1
            if valid_indices is None:
                valid_indices = not_cysteine_indices
            else:
                valid_indices = np.intersect1d(
                    valid_indices, not_cysteine_indices, assume_unique=True
                )

        self.sequences = []
        self.likelihood = []
        self.mutants = []

        mutants = {}
        likelihood = self.compute_likelihood(sequence)
        self._add(sequence, likelihood, mutants)

        for step in trange(self.steps):
            new_sequence, new_mutants = self._make_move(
                sequence, mutants, valid_indices=valid_indices
            )
            likelihood = self.compute_likelihood(new_sequence)
            if self._accept(likelihood):
                sequence = new_sequence
                mutants = new_mutants
                self._add(sequence, likelihood, mutants)
        result = pd.DataFrame(
            {
                "sequence": self.sequences,
                "likelihood": self.likelihood,
                "mutant": self.mutants,
            }
        )

        delattr(self, "sequences")
        delattr(self, "likelihood")
        delattr(self, "mutants")
        return result


class UnsupervisedESM1vMCMC(MCMC):
    _MODEL = None
    _ALPHABET = None
    _VOCAB = None

    def __init__(
        self,
        steps: int = 10000,
        max_mutants: int = 3,
        no_cysteine: bool = False,
        mask_positions: bool = False,
    ):
        super().__init__(steps, max_mutants, no_cysteine)
        self.mask_positions = mask_positions

    def get_model(self) -> Tuple[esm.model.ProteinBertModel, esm.data.Alphabet, Vocab]:
        if self._MODEL is None:
            print("Initializing Model")
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
            model = model.eval().cuda().requires_grad_(False)
            vocab = Vocab.from_esm_alphabet(alphabet)
            self._MODEL = model
            self._ALPHABET = alphabet
            self._VOCAB = vocab
        return self._MODEL, self._ALPHABET, self._VOCAB

    def compute_likelihood(self, seq: str) -> float:
        model, alphabet, vocab = self.get_model()
        neg_logp = sequence_pseudo_ppl(
            model, vocab, seq, mask_positions=self.mask_positions, reduction="sum", log=True
        )
        return math.exp(-neg_logp)


class SupervisedESM1bMCMC(MCMC):
    _MODEL = None

    def get_model(self):
        if self._MODEL is None:
            sys.path.append(str(Path(__file__).parents[1] / "supervised-stability"))
            from inference import EnsemblePredictionModel

            self._MODEL = EnsemblePredictionModel()
        return self._MODEL

    def compute_likelihood(self, seq: str) -> float:
        model = self.get_model()
        return model.probability(model.predict(seq))
