from typing import Union
from pathlib import Path
import numpy as np
import torch
import yaml
import boto3
import pandas as pd
from sklearn.neighbors import KernelDensity
from supervised import StabilityModel
from dataset import FeatureType, FeatureTypeStrategy


class EnsemblePredictionModel:
    """Model that performs inference on an input sequence."""

    def __init__(
        self,
        features: str = "esm1b",
        head: str = "concat",
        mlp: bool = True,
        regression: bool = False,
        singer: bool = False,
        checkpoint_path: Union[str, Path] = Path(__file__).parents[1] / "logs",
        output_path: Union[str, Path] = Path(__file__).parents[1] / "outputs",
    ):
        self.features = FeatureType[features.upper()]
        self.head = head
        self.mlp = mlp
        self.regression = regression
        self.singer = singer
        self.output_path = Path(output_path)
        self.strategy = FeatureTypeStrategy(self.features)

        self.name = self.make_name()
        checkpoints = (Path(checkpoint_path) / self.name).rglob("*.ckpt")
        self.head_models = [
            StabilityModel.load_from_checkpoint(str(ckpt)).eval().cuda()
            for ckpt in checkpoints
        ]
        assert self.head_models, f"No matching checkpoints found: {self.name}"

        self.base_model = self.strategy.build_model()
        if not self.features.is_unirep:
            self.base_model = self.base_model.cuda()
        self.vocab = self.strategy.build_vocab()

        # obtained by finding average TS50 value of sequences in each of four bins
        # bins are <50, 50-60, 60-70, >70
        self.bin_values = torch.tensor(
            [36.0395, 56.6874, 64.190, 70.0740], device="cuda"
        )

        self.output_path.mkdir(exist_ok=True)
        ts50_output_path = self.output_path / f"ts50_{self.make_name(remove_nonvariable=True)}.csv"
        predictions = pd.read_csv(ts50_output_path)
        self.kde = KernelDensity(kernel="gaussian")
        self.kde.fit(np.asarray(predictions["E[TS50]"])[:, None])


    def make_name(
        self,
        remove_nonvariable: bool = False,
    ):
        name = []
        name.append(f"base{self.features.value.lower()}")
        name.append(f"head{self.head}")
        if self.mlp:
            name.append("mlp")
        if self.regression:
            name.append("regression")
        if self.singer:
            name.append("singer")
        if not remove_nonvariable:
            name.append("seed0_lr0.001_batch64")
        return "_".join(name)

    def predict(self, sequence: str):
        if self.features.is_unirep:
            reps = self.strategy.forward(self.base_model, [sequence]).cuda()
        else:
            tokens = torch.from_numpy(self.vocab.encode(sequence)).cuda().unsqueeze(0)
            reps = self.strategy.forward(self.base_model, tokens)
        predictions = torch.stack(
            [model(features=reps)["prediction"] for model in self.head_models], 0
        ).mean(0)

        if self.regression:
            return predictions.item()
        else:
            probs = predictions.softmax(-1)
            expected_value = probs @ self.bin_values
            return expected_value.item()

    def probability(self, prediction: float) -> float:
        return np.exp(self.kde.score(np.array([[prediction]])))
