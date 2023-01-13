from typing import List
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm
import esm
from evo.likelihood import pseudo_ppl


class UnsupervisedESM1vScorer:
    def __init__(self, mask_positions=False):
        self.mask_positions = mask_positions
        self.model_names = [f"esm1v_t33_650M_UR90S_{i}" for i in range(1, 6)]
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")

    def score(self, sequences: List[str]) -> np.ndarray:
        all_scores = []
        for name in self.model_names:
            model, _ = esm.pretrained.load_model_and_alphabet(name)
            model = model.eval().cuda().requires_grad_(False)
            scores = pseudo_ppl(
                model, self.alphabet, sequences, mask_positions=self.mask_positions, log=True,
            )
            all_scores.append(scores.cpu())

        return torch.stack(all_scores, 0).mean(0).numpy()


class SupervisedESM1bScorer:
    def __init__(self):
        sys.path.append(str(Path(__file__).parents[1] / "supervised-stability"))
        from inference import EnsemblePredictionModel
        self.model = EnsemblePredictionModel()

    def score(self, sequences: List[str]) -> np.ndarray:
        return np.array([self.compute_likelihood(seq) for seq in tqdm(sequences)])

    def compute_likelihood(self, seq: str) -> float:
        return self.model.probability(self.model.predict(seq))
