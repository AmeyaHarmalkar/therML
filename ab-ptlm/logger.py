# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""
import pickle as pkl
import logging
import os
from typing import Dict, Optional, Union
from pathlib import Path

import torch
import numpy as np

from pytorch_lightning.loggers.base import rank_zero_experiment
from pytorch_lightning.loggers.csv_logs import ExperimentWriter, CSVLogger

log = logging.getLogger(__name__)



class CustomExperimentWriter(ExperimentWriter):
    r"""
    Experiment writer for CSVLogger.

    Currently supports to log hyperparameters and metrics in YAML and CSV
    format, respectively.

    Args:
        log_dir: Directory for the experiment logs
    """
    NAME_PREDICTIONS_FILE = 'predictions.pkl'
    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)
        self.predictions_file_path = Path(self.log_dir) / self.NAME_PREDICTIONS_FILE

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics"""

        def _handle_value(value):
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)
        metrics = {}
        for key, value in metrics_dict.items():
            if "/" in key:
                split, name = key.split("/")
                if "split" in metrics:
                    assert metrics["split"] == split
                else:
                    metrics["split"] = split
                metrics[name] = _handle_value(value)
        # metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics['step'] = step
        self.metrics.append(metrics)

    def log_predictions(self, predictions) -> None:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu()
        elif isinstance(predictions, (list, tuple)):
            predictions = [p.cpu() if isinstance(p, torch.tensor) else p for p in predictions]
        elif isinstance(predictions, dict):
            predictions = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in predictions.items()}
        self.predictions = predictions

    def save(self):
        super().save()
        if hasattr(self, "predictions"):
            with open(self.predictions_file_path, "wb") as f:
                pkl.dump(self.predictions, f)

class CustomCSVLogger(CSVLogger):
    r"""
    Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.loggers import CSVLogger
        >>> logger = CSVLogger("logs", name="my_exp_name")
        >>> trainer = Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name. Defaults to ``'default'``.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
    """

    @property
    @rank_zero_experiment
    def experiment(self) -> CustomExperimentWriter:
        r"""

        Actual ExperimentWriter object. To use ExperimentWriter features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment:
            return self._experiment

        os.makedirs(self.root_dir, exist_ok=True)
        self._experiment = CustomExperimentWriter(log_dir=self.log_dir)
        return self._experiment