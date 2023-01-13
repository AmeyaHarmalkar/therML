from typing import Sequence, Union, Optional, Dict
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from dataset import FeatureType, FeatureTypeStrategy
from supervised import StabilityModel
from functools import partial


def make_name(
    features: FeatureType,
    head: str,
    mlp: bool,
    regression: bool,
    singer: bool = False,
    remove_nonvariable: bool = False,
):
    name = []
    name.append(f"base{features.value.lower()}")
    name.append(f"head{head}")
    if mlp:
        name.append("mlp")
    if regression:
        name.append("regression")
    if singer:
        name.append("singer")
    if not remove_nonvariable:
        name.append("seed0_lr0.001_batch64")
    return "_".join(name)


def postprocess(output: torch.Tensor, regression: bool = False):
    if not regression:
        bin_values = torch.tensor([36.0395, 56.6874, 64.190, 70.0740], device="cuda")
        p = output.softmax(-1)
        return (p @ bin_values).cpu()
    else:
        return output.cpu()


@torch.no_grad()
def predict(
    sequences: Sequence[str],
    head: str = "concat",
    features: Union[str, FeatureType] = FeatureType.ESM1B,
    mlp: bool = False,
    regression: bool = False,
    singer: bool = False,
) -> torch.Tensor:
    if isinstance(features, str):
        features = FeatureType[features.upper()]
    # FeatureTypeStrategy has methods for building vocab/models for each feature type
    strategy = FeatureTypeStrategy(features)

    # Create the feature extraction model
    base_model = strategy.build_model()
    if not features.is_unirep:
        base_model = base_model.cuda()
    vocab = strategy.build_vocab()

    # Try to load checkpoints that match the hyperparameters specified
    name = make_name(features, head, mlp, regression, singer)
    checkpoints = (Path(__file__).parents[1] / "logs" / name).rglob("*.ckpt")
    models = [
        StabilityModel.load_from_checkpoint(str(ckpt)).eval().cuda()
        for ckpt in checkpoints
    ]
    assert models, f"No matching checkpoints found: {name}"

    process = partial(postprocess, regression=regression)

    if features.is_unirep:
        # If UniRep, features are extracted with JAX code, batch predicted at the end
        reps = strategy.forward(base_model, sequences).cuda()
        predictions = torch.stack(
            [model(features=reps)["prediction"] for model in models], 0
        ).mean(0)
        exp_ts50 = process(predictions)
        output = {"E[TS50]": exp_ts50.numpy()}
    else:
        # Otherwise make the predictions one-by-one
        # Output should have predictions, expected TS50, and potentially the embedding vectors
        exp_ts50 = []
        embeddings = []
        for sequence in tqdm(sequences):
            tokens = torch.from_numpy(vocab.encode(sequence)).cuda().unsqueeze(0)
            reps = strategy.forward(base_model, tokens)
            preds = []
            embeds = []
            for model in models:
                output = model(features=reps, return_embedding=True)
                preds.append(output["prediction"])
                if "embed" in output:
                    embeds.append(output["embed"])
            predictions = torch.cat(preds, 0).mean(0)
            exp_ts50.append(process(predictions))
            if embeds:
                embeds = torch.cat(embeds, 0)
                embeddings.append(embeds.cpu())
        exp_ts50 = torch.stack(exp_ts50, 0)
        output = {"E[TS50]": exp_ts50.numpy()}
        if embeddings:
            embeddings = torch.stack(embeddings, 0)
            output["embed"] = embeddings.numpy()
    return output


def corr(df: pd.DataFrame, key: str, mask: Optional[pd.Series] = None) -> float:
    """Computer correlations"""
    if mask is not None:
        df = df[mask]
    return df["E[TS50]"].corr(df[key], method="spearman")


class EvaluationDataset:
    """ A simple helper class for each of the downstream stability evaluation datasets.
    """
    def __init__(
        self,
        filename: str,
        name: str,
        sequence_key: str,
        measurement_key: str,
        can_split_with_linker: bool = False,
    ):
        filepath = Path(__file__).parents[1] / "data" / filename
        self.data = pd.read_csv(filepath)
        self.name = name
        self.sequence_key = sequence_key
        self.measurement_key = measurement_key
        self.can_split_with_linker = can_split_with_linker
        self.splits = {}

        if not pd.api.types.is_numeric_dtype(self.measurements):
            self.data.loc[self.measurements == "up", self.measurement_key] = 70
            self.data.loc[self.measurements == "negativ", self.measurement_key] = float(
                "nan"
            )
            self.data[self.measurement_key] = self.measurements.astype(float)

    @property
    def sequences(self):
        return self.data[self.sequence_key]

    @property
    def measurements(self):
        return self.data[self.measurement_key]

    @property
    def predictions(self):
        return self.data["E[TS50]"]

    @property
    def has_predictions(self) -> bool:
        return "E[TS50]" in self.data.columns

    @property
    def has_embed(self) -> bool:
        return hasattr(self, "embeds")

    def __len__(self):
        return len(self.data)

    def add_predictions(self, predictions: Sequence[float]) -> None:
        self.data["E[TS50]"] = predictions

    def add_embeds(self, embeds: np.ndarray) -> None:
        self.embeds = embeds

    def save_predictions(self, method: str) -> None:
        filepath = Path(__file__).parents[1] / "outputs"/ f"{self.name}_{method}.csv"
        self.data.to_csv(filepath, index=False)

    def correlation(self) -> Dict[str, float]:
        metrics = {
            f"{self.name}_corr_all": self.predictions.corr(
                self.measurements, method="spearman"
            )
        }
        for split, mask in self.splits.items():
            metrics[f"{self.name}_corr_{split}"] = self.predictions[mask].corr(
                self.measurements[mask], method="spearman"
            )
        return metrics

    def add_splits(self, **splits):
        self.splits.update(splits)


def create_datasets(
    names: Optional[Sequence[str]] = None,
) -> Dict[str, EvaluationDataset]:
    if names is None:
        names = {
            "ts50",
            "bioreg_cepa",
            "bioreg_iso",
            "bite",
            "protherm",
            "i2c",
            "MUC13",
        }
    datasets = {}
    if "ts50" in names:
        datasets["ts50"] = EvaluationDataset(
            "ts50_all.csv", "ts50", "Sequence", "TS50_float", can_split_with_linker=True
        )
    if "bioreg_cepa" in names:
        bioreg_cepa = EvaluationDataset(
            "ALL_bioreg_scFvs_iso_cepa.csv",
            "bioreg_cepa",
            "PROTSEQFULL_CHAIN",
            "cepa_Tm1",
        )
        bioreg_cepa.add_splits(
            heavy=bioreg_cepa.data["start"] == "heavy",
            light=bioreg_cepa.data["start"] == "light",
        )
        datasets["bioreg_cepa"] = bioreg_cepa
    if "bioreg_iso" in names:
        bioreg_iso = EvaluationDataset(
            "ALL_bioreg_scFvs_iso_cepa.csv",
            "bioreg_iso",
            "PROTSEQFULL_CHAIN",
            "iso_Tm1",
        )
        bioreg_iso.add_splits(
            heavy=bioreg_iso.data["start"] == "heavy",
            light=bioreg_iso.data["start"] == "light",
        )
        datasets["bioreg_iso"] = bioreg_iso
    if "bite" in names:
        datasets["bite"] = EvaluationDataset(
            "BiTE_Tm.csv", "bite", "sequence", "Tm", can_split_with_linker=True
        )
    if "protherm" in names:
        datasets["protherm"] = EvaluationDataset(
            "protherm_Tm.csv",
            "protherm",
            "sequence",
            "dTm",
            can_split_with_linker=False,
        )
    if "i2c" in names:
        datasets["i2c"] = EvaluationDataset(
            "I2Copt_ts50.csv", "i2c", "sequence", "TS50(°C)", can_split_with_linker=True
        )
    if "MUC13" in names:
        datasets["MUC13"] = EvaluationDataset(
            "MUC13-singlepointscan.csv",
            "MUC13",
            "sequence",
            "Total",
            can_split_with_linker=True,
        )
    return datasets


def predict_all(
    head: str = "concat",
    features: Union[str, FeatureType] = FeatureType.ESM1B,
    mlp: bool = False,
    regression: bool = False,
    singer: bool = False,
    names: Optional[Sequence[str]] = None,
):
    if isinstance(features, str):
        features = FeatureType[features.upper()]

    method = make_name(features, head, mlp, regression, singer, remove_nonvariable=True)
    datasets = create_datasets(names)

    pred = partial(
        predict,
        head=head,
        features=features,
        mlp=mlp,
        regression=regression,
        singer=singer,
    )
    metrics = {}
    for name, dataset in datasets.items():
        if features == FeatureType.DEEPAB and not dataset.can_split_with_linker:
            continue
        result = pred(dataset.sequences)
        dataset.add_predictions(result["E[TS50]"])
        if "embed" in result:
            dataset.add_embeds(result["embed"])
        dataset.save_predictions(method)
        corrs = dataset.correlation()
        metrics.update(corrs)
    metrics = pd.Series(metrics, name=method)
    print(metrics)
    if not args.names:
        # Uncomment to save all outputs
        # np.savez(
        # f"/home/rrao04/projects/stability/outputs/{method}_embed.npz",
        # embed=np.concatenate([dataset.embeds for dataset in datasets.values() if dataset.has_embed]),
        # measurement=np.concatenate([dataset.measurements for dataset in datasets.values() if dataset.has_predictions]),
        # prediction=np.concatenate([dataset.predictions for dataset in datasets.values() if dataset.has_predictions]),
        # dataset=list(itertools.chain.from_iterable(
        # [dataset.name] * len(dataset) for dataset in datasets.values() if dataset.has_predictions
        # ))
        # )
        outfile = Path(__file__).parents[1] / "outputs" / "correlations.csv")
        if outfile.exists():
            all_metrics = pd.read_csv(outfile).set_index("Method")
            all_metrics.loc[method] = metrics
        else:
            all_metrics = pd.DataFrame([metrics])
            all_metrics.index.name = "Method"
        all_metrics = all_metrics.reset_index()
        all_metrics.to_csv(outfile, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="esm1b")
    parser.add_argument("--head_model", default="concat")
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--regression", action="store_true")
    parser.add_argument("--singer", action="store_true")
    parser.add_argument("--names", nargs="*", default=None)
    args = parser.parse_args()
    predict_all(
        args.head_model,
        args.base_model,
        args.mlp,
        args.regression,
        args.singer,
        names=args.names,
    )
