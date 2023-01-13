from asyncio.log import logger
from typing import Optional, Any
import itertools
from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torchmetrics
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from logger import CustomCSVLogger
from head_models import (
    HeadType,
    AttentionWeightedMean,
    AttentionPooling,
    ConcatProject,
    OutputHead,
    TRRosettaHead,
)
from dataset import (
    Mode,
    FeatureType,
    FeatureTypeStrategy,
    StabilityDataset,
)

LOG_DIR = Path(__file__).absolute().parent.with_name("logs")


class Accuracy(torchmetrics.Metric):
    def __init__(self, ignore_index: int = -1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets. See :ref:`references/modules:input types` for more information
        on input types.
        Args:
            preds: Predictions from model (logits, probabilities, or labels)
            target: Ground truth labels
        """
        mask = target != self.ignore_index
        values = preds[mask].argmax(-1)
        target = target[mask]
        self.correct += (values == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total



class FeatureModel(nn.Module):
    def __init__(self, feature_strategy: FeatureTypeStrategy, inference: bool = False):
        super().__init__()
        self.feature_strategy = feature_strategy
        self.inference = inference
        if inference:
            self.model = self.feature_strategy.build_model()

    def forward(self, tokens, features=None):
        if self.inference and features is None:
            return self.feature_strategy.forward(self.model, tokens)
        else:
            return features

    @property
    def output_dim(self) -> int:
        return self.feature_strategy.output_dim()


@dataclass
class SupervisedModelConfig:
    base_model: str = "antiberty"
    head_model: str = "concat"
    mlp: bool = False
    learning_rate: float = 1e-3
    batch_size: int = 32
    paired: bool = False
    max_length: Optional[int] = None
    inference: bool = False
    regression: bool = False


class StabilityModel( pl.LightningModule ):

    def __init__( self, config: SupervisedModelConfig = SupervisedModelConfig(), **hparams):
        super().__init__()
        self.save_hyperparameters(vars(config))
        self.save_hyperparameters(hparams)

        feature_type = FeatureType[ self.hparams.base_model.upper() ]
        self.feature_strategy = FeatureTypeStrategy( feature_type, "ts50" )
        self.base_model = FeatureModel( self.feature_strategy, self.hparams.inference )
        in_dim = self.base_model.output_dim

        head_type = HeadType[ self.hparams.head_model.upper() ]

        if feature_type.is_single_vector:
            self.head = nn.Identity()
            out_dim = self.feature_strategy.output_dim()
        elif head_type == HeadType.ATTNMEAN:
            self.head = AttentionWeightedMean(in_dim, dropout=0.1)
            out_dim = in_dim
        elif head_type == HeadType.CONCAT:
            self.head = ConcatProject(in_dim, 4, self.hparams.max_length)
            out_dim = 4 * self.hparams.max_length
        
        self.output = self.build_output( out_dim )

        self.accuracy = nn.ModuleDict(
            {mode.value : Accuracy( ignore_index = -1 ) for mode in Mode }
        )

    
    def build_output(self, out_dim: int):

        proj_dim = 4 if not self.hparams.regression else 1
        return OutputHead(
            out_dim,
            proj_dim,
            mlp = self.hparams.mlp
        )
    

    def forward(
        self,
        tokens=None,
        features=None,
        return_embedding: bool = True,
        **unused,
    ):
        output = {}

        if tokens is not None or features is not None:
            embed = self.head(self.base_model(tokens, features))
            return_embedding = return_embedding and self.hparams.mlp
            prediction = self.output(embed, return_embedding=return_embedding)
            if return_embedding:
                prediction, embed = prediction
                output["embed"] = embed
            output["prediction"] = prediction
        assert output, "No input passed to model"
        return output

    
    def compute_and_log_loss( self, batch, mode: Mode ):
        output = self(**batch)
        loss = 0
        if "prediction" in output:
            prediction = output["prediction"]
            if not self.hparams.regression:
                label = batch["label"]
                ts50_loss = nn.CrossEntropyLoss( ignore_index=-1 )(prediction, label)
                self.log(
                    f"{mode.value.lower()}/acc",
                    self.accuracy[mode.value](prediction, label),
                )
            else:
                label = batch["ts50"]
                ts50_loss = nn.SmoothL1Loss()(prediction, label)
            loss += ts50_loss
            self.log( f"{mode.value.lower()}/ts50", ts50_loss )

            output = {
                "label" : label,
                "prediction" : prediction,
                "name" : batch["name"],
            }
        self.log( f"{mode.value.lower()}/loss", loss )
        return loss, output

    
    def training_step(self, batch, batch_idx):
        return self.compute_and_log_loss(batch, Mode.TRAIN)[0]

    def validation_step(self, batch, batch_idx):
        return self.compute_and_log_loss(batch, Mode.VALID)[0]

    def test_step(self, batch, batch_idx):
        _, prediction = self.compute_and_log_loss(batch, Mode.TEST)
        return prediction

    def test_epoch_end( self, outputs ):
        all_outputs = {}
        for key in outputs[0]:
            if isinstance(outputs[0][key], torch.Tensor):
                value = torch.cat([el[key] for el in outputs])
            else:
                value = list(itertools.chain.from_iterable([el[key] for el in outputs]))
            all_outputs[key] = value
        self.logger.experiment.log_predictions(all_outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        return optimizer

    def state_dict(self):
        out = {
            k: v
            for k, v in super().state_dict().items()
            if not k.startswith("base_model")
        }
        return out
        

class PairedStabilityModel( StabilityModel ):
    
    def build_output( self, out_dim: int ):
        return OutputHead(
            2 * out_dim,
            7,
            mlp = self.hparams.mlp,
        )
    
    def forward( self, tokens1=None, features1=None, tokens2=None, features2=None, **unused):
        embed1 = self.head(self.base_model(tokens1, features1))
        embed2 = self.head(self.base_model(tokens2, features2))
        embed_both = torch.cat([embed1, embed2], 1)
        prediction = self.output(embed_both)
        return prediction

    
    def compute_and_log_loss(self, batch, mode: Mode):
        b1, b2, label = batch
        prediction = self(b1["tokens"], b1["features"], b2["tokens"], b2["features"])
        output = {
            "diff_label": label,
            "diff_pred": prediction,
            "first_name": b1["name"],
            "second_name": b2["name"],
        }
        loss = nn.CrossEntropyLoss(ignore_index=-1)(prediction, label)
        self.log(f"{mode.value.lower()}/loss", loss)
        self.log(
            f"{mode.value.lower()}/acc", self.accuracy[mode.value](prediction, label)
        )
        return loss, output

    
    @property
    def out_bins(self) -> int:
        return 7

    @property
    def max_length(self) -> int:
        return 2 * self.hparams.max_length
    

def make_name(args) -> str:
    name = []
    if args.paired:
        name.append("paired")
    name.append(f"base{args.base_model}")
    name.append(f"head{args.head_model}")
    if args.mlp:
        name.append("mlp")
    if args.regression:
        name.append("regression")
    name.append(f"seed{args.seed}")
    name.append(f"lr{args.learning_rate}")
    name.append(f"batch{args.batch_size}")
    name.append("epochtest")
    return "_".join(name)


def train(args, train_args):
    feature_type = FeatureType[ args.base_model.upper() ]
    feature_strategy = FeatureTypeStrategy(feature_type, "ts50")
    data = {
        mode: StabilityDataset(
            split = args.split,
            mode=mode,
            seed=args.seed,
            feature_strategy=feature_strategy,
        )
        for mode in Mode
    }

    max_len = max(d.max_length for d in data.values())
    split = data[Mode.TRAIN].split

    logger = CustomCSVLogger(
        LOG_DIR,
        name=make_name(args),
        version=split,
    )

    if not args.override and logger.experiment.predictions_file_path.exists():
        print(
            "Already have predictions for this run, delete them to re-train"
        )
        return

    loaders = {
        mode: DataLoader(
            dataset,
            num_workers=args.num_workers,
            collate_fn=dataset.collater,
            batch_size=args.batch_size,
            shuffle=mode == Mode.TRAIN,
        )
        for mode, dataset in data.items()
    }

    config = SupervisedModelConfig(
        base_model=args.base_model,
        head_model=args.head_model,
        mlp=args.mlp,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        paired=args.paired,
        regression=args.regression,
        max_length=max_len,
    )

    model = (PairedStabilityModel if args.paired else StabilityModel)(config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="valid/loss",
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="valid/loss",
        patience=5,
    )

    trainer = pl.Trainer.from_argparse_args(
        train_args,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        gpus=1,
        max_epochs=25,
    )

    if not args.eval:
        trainer.fit(
            model,
            train_dataloaders=loaders[Mode.TRAIN],
            val_dataloaders=loaders[Mode.VALID],
        )
    
    trainer.test(model, dataloaders=loaders[Mode.TEST])


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="0", choices=list(map(str, range(16))) + ["all"]
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="antiberty",
        choices=[f.value.lower() for f in FeatureType],
    )
    parser.add_argument(
        "--head_model",
        type=str,
        default="concat",
        choices=[h.value.lower() for h in HeadType],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mlp", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--eval", action="store_true")
    #parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--regression", action="store_true")
    train_parser = pl.Trainer.add_argparse_args(parser)
    args, _ = parser.parse_known_args()
    train_args = train_parser.parse_args()
    logging.getLogger("filelock").setLevel(logging.ERROR)

    if args.split == "all":

        from evo.distribute import poll_gpu_with_commands

        def pair_to_arg(key: str, value: Any) -> str:
            if isinstance(value, bool):
                return f"--{key}" if value else ""
            elif value is not None:
                return f"--{key} {value}"
            else:
                return ""

        def commands():
            for split in range(17):
                command = f"python supervised.py --split {split} " + " ".join(
                    pair_to_arg(key, value)
                    for key, value in vars(args).items()
                    if key != "split" and value is not None
                )
                yield command.split()

        poll_gpu_with_commands(commands())
    else:
        if args.split.isdecimal():
            args.split = int(args.split)
        train(args, train_args)




        

    


            
