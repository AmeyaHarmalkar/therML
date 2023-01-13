from typing import Union, Tuple, Dict, List, Any
import tarfile
import enum
import hashlib
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import esm
from evo.tensor import numpy_seed, collate_tensors
from evo.tokenization import Vocab
from tape import TRRosetta
from filelock import FileLock


DATA_DIR = Path(__file__).absolute().parent.with_name("data")


def collate_tokens(tokens, vocab: Vocab):
    if tokens[0].dtype == torch.long:
        tokens = collate_tensors(tokens, constant_value=vocab.pad_idx)
    else:
        tokens = collate_tensors(tokens, constant_value=0)
    return tokens


def hash_sequence(seq: str, prefix: str = "") -> str:
    return hashlib.md5(f"{prefix}{seq}".encode()).hexdigest()


class FeatureType(enum.Enum):
    ESM1B = "ESM1B"
    ESMMSA = "ESMMSA"
    TRROSETTA = "TRROSETTA"
    ROSETTAFOLD = "ROSETTAFOLD"
    UNIREP = "UNIREP"
    UNIREP_EVOTUNE_MSA = "UNIREP_EVOTUNE_MSA"
    UNIREP_EVOTUNE_OAS = "UNIREP_EVOTUNE_OAS"
    DEEPAB = "DEEPAB"

    @property
    def is_unirep(self) -> bool:
        return self in (
            FeatureType.UNIREP,
            FeatureType.UNIREP_EVOTUNE_MSA,
            FeatureType.UNIREP_EVOTUNE_OAS,
        )

    @property
    def is_deepab(self) -> bool:
        return self == FeatureType.DEEPAB

    @property
    def is_single_vector(self) -> bool:
        return self.is_unirep or self.is_deepab


class Mode(enum.Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class FeatureTypeStrategy:

    _BUILT_MODEL = None

    FEATURE_DIRS = {
        FeatureType.ESM1B: "esm1b",
        FeatureType.ESMMSA: "esm_msa",
        FeatureType.TRROSETTA: "trrosetta",
        FeatureType.ROSETTAFOLD: "rosettafold",
        FeatureType.UNIREP: "unirep.npz",
        FeatureType.UNIREP_EVOTUNE_MSA: "unirep_evotune_msa.npz",
        FeatureType.UNIREP_EVOTUNE_OAS: "unirep_evotune_oas.npz",
        FeatureType.DEEPAB: "deepab.npy",
    }

    SINGER_FEATURE_DIRS = {
        FeatureType.ESM1B: DATA_DIR / "singer_esm.lmdb",
        FeatureType.UNIREP: DATA_DIR / "singer_unirep_sharded.npz",
    }

    def __init__(self, feature_type: FeatureType, dataset: str = "ts50"):
        self.dataset = dataset
        self.feature_type = feature_type
        self.feature_dir = DATA_DIR / (f"{dataset}_" + self.FEATURE_DIRS[self.feature_type])
        self.singer_feature_dir = self.SINGER_FEATURE_DIRS.get(self.feature_type, None)

    def maybe_download(self):
        if not self.feature_dir.exists():
            print("Downloading features")
            import boto3

            s3 = boto3.client("s3")
            if self.feature_dir.suffix in (".npz", ".npy"):
                s3.download_file(
                    "stability-prediction", self.feature_dir.name, str(self.feature_dir)
                )
            else:
                tarpath = self.feature_dir.with_suffix(".tar.gz")
                tarpathlock = self.feature_dir.with_suffix(".lock")
                with FileLock(tarpathlock):
                    if self.feature_dir.exists():
                        return

                    self.feature_dir.parent.mkdir(exist_ok=True)
                    s3.download_file("stability-prediction", tarpath.name, str(tarpath))
                    with tarfile.open(tarpath) as f:
                        f.extractall(tarpath.parent)

    def maybe_download_singer(self):
        if self.singer_feature_dir and not self.singer_feature_dir.exists():
            print("Downloading features")
            import boto3

            s3 = boto3.client("s3")
            if self.singer_feature_dir.suffix in (".npy", ".lmdb"):
                s3.download_file(
                    "stability-prediction",
                    self.singer_feature_dir.name,
                    str(self.singer_feature_dir),
                )
                s3.download_file(
                    "stability-prediction",
                    "singer_unirep_index.npy",
                    str(self.singer_feature_dir.with_name("singer_unirep_index.npy")),
                )
            else:
                tarpath = self.feature_dir.with_suffix(".tar.gz")
                tarpathlock = self.feature_dir.with_suffix(".lock")
                with FileLock(tarpathlock):
                    if self.feature_dir.exists():
                        return

                    self.feature_dir.parent.mkdir(exist_ok=True)
                    s3.download_file("stability-prediction", tarpath.name, str(tarpath))
                    with tarfile.open(tarpath) as f:
                        f.extractall(tarpath.parent)

    def build_model(self):
        if self._BUILT_MODEL is not None:
            return self._BUILT_MODEL
        if self.feature_type == FeatureType.ESM1B:
            model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
        elif self.feature_type == FeatureType.ESMMSA:
            model, _ = esm.pretrained.esm_msa1_t12_100M_UR50S()
        elif self.feature_type == FeatureType.TRROSETTA:
            model = TRRosetta.from_pretrained("xaa")
        elif self.feature_type.is_unirep:
            import os

            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
            import jax_unirep

            params = {
                FeatureType.UNIREP: lambda: None,
                FeatureType.UNIREP_EVOTUNE_MSA: lambda: np.load(
                    DATA_DIR / "unirep_evotune_ts50_msa.npz"
                ),
                FeatureType.UNIREP_EVOTUNE_OAS: lambda: np.load(
                    DATA_DIR / "unirep_evotune_paired_oas.npz"
                ),
            }[self.feature_type]()

            vocab = self.build_vocab()

            def model(tokens):
                if isinstance(tokens, torch.Tensor):
                    sequences = vocab.decode(tokens.cpu().numpy())
                    device = tokens.device
                else:
                    sequences = tokens
                    device = "cpu"
                h_avg, h_final, c_final = jax_unirep.get_reps(sequences, params=params)
                rep = np.concatenate([h_avg, h_final], -1)
                return torch.from_numpy(rep).float().to(device)

            return model
        elif self.feature_type.is_deepab:
            import sys

            sys.path.append("/home/rrao04/projects/deepab")
            from deepab.models.ModelEnsemble import ModelEnsemble
            from deepab.models.AbResNet import load_model

            model_files = list(
                (
                    Path(__file__).absolute().parents[2]
                    / "deepab"
                    / "trained_models"
                    / "ensemble_abresnet"
                ).glob("*.pt")
            )
            assert model_files, "No model files found"
            model = ModelEnsemble(
                model_files=model_files,
                load_model=load_model,
                eval_mode=True,
                device="cuda",
            )

        else:
            raise NotImplementedError(self.feature_type)
        model = model.eval().requires_grad_(False)
        self.__class__._BUILT_MODEL = model
        return model

    def forward(self, model, tokens):
        if self.feature_type == FeatureType.ESM1B:
            return model(tokens, repr_layers=[33])["representations"][33][:, 1:-1]
        elif self.feature_type == FeatureType.ESMMSA:
            return model(tokens, repr_layers=[12])["representations"][12][:, 0, 1:]
        elif self.feature_type == FeatureType.TRROSETTA:
            return model(F.one_hot(tokens).float())[-1]
        elif self.feature_type.is_unirep:
            return model(tokens)
        elif self.feature_type.is_deepab:
            encoding = []
            for model_ in model.models:
                lstm_input, input_delims = model_.get_lstm_input(tokens)
                _, (enc, _) = model_.lstm_model.encoder(src=lstm_input)
                encoding.append(enc)
            return torch.stack(encoding, 0).mean(0)
        else:
            raise NotImplementedError(self.feature_type)

    def output_dim(self) -> int:
        if self.feature_type == FeatureType.ESM1B:
            return 1280
        elif self.feature_type == FeatureType.ESMMSA:
            return 768
        elif self.feature_type == FeatureType.TRROSETTA:
            return 1024
        elif self.feature_type == FeatureType.ROSETTAFOLD:
            return 384
        elif self.feature_type.is_unirep:
            return 3800
        elif self.feature_type.is_deepab:
            return 64
        else:
            raise NotImplementedError(self.feature_type)

    def build_vocab(self):
        if self.feature_type == FeatureType.ESM1B:
            alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            return Vocab.from_esm_alphabet(alphabet)
        elif self.feature_type == FeatureType.ESMMSA:
            alphabet = esm.data.Alphabet.from_architecture("MSA Transformer")
            return Vocab.from_esm_alphabet(alphabet)
        elif self.feature_type in (FeatureType.TRROSETTA, FeatureType.ROSETTAFOLD):
            return Vocab.from_trrosetta()
        elif self.feature_type.is_unirep:
            from tape.tokenizers import UNIREP_VOCAB

            return Vocab(UNIREP_VOCAB)
        elif self.feature_type.is_deepab:
            from deepab_vocab import DeepabVocab

            return DeepabVocab()
        else:
            raise NotImplementedError(self.feature_type)

    def item_to_feature_path(self, item: pd.Series) -> Path:
        if self.feature_type == FeatureType.ESM1B:
            if self.dataset == "ts50":
                return self.feature_dir / f"ts50_{item.name}.pt"
            else:
                return self.feature_dir / f"{item.name}.pt"
        elif self.feature_type == FeatureType.ROSETTAFOLD:
            return self.feature_dir / f"{item['Name']}.npz"
        elif self.feature_type.is_unirep:
            return item.name
        elif self.feature_type.is_deepab:
            return item.name
        else:
            return self.feature_dir / f"{item['Name']}.pt"

    def load_features(self, item: pd.Series) -> torch.Tensor:
        path = self.item_to_feature_path(item)

        if self.feature_type == FeatureType.ROSETTAFOLD:
            rep = np.load(path)["embed"]
            return torch.from_numpy(rep).float()
        elif self.feature_type.is_unirep:
            if not hasattr(self, "_unirep_data"):
                data = np.load(self.feature_dir)
                self._unirep_data = np.concatenate((data["h_avg"], data["h_final"]), -1)
            return torch.from_numpy(self._unirep_data[path]).float()
        elif self.feature_type.is_deepab:
            if not hasattr(self, "_deepab_data"):
                self._deepab_data = np.load(self.feature_dir)
            return torch.from_numpy(self._deepab_data[path]).float()

        rep = torch.load(path)
        if self.feature_type == FeatureType.ESM1B:
            return rep["representations"][33]
        elif self.feature_type == FeatureType.ESMMSA:
            return rep["representations"][12].float()
        else:
            return rep

    def load_singer_features(self, item: pd.Series) -> torch.Tensor:
        if self.feature_type == FeatureType.ROSETTAFOLD:
            raise NotImplementedError
        elif self.feature_type.is_unirep:
            if not hasattr(self, "_unirep_singer_data"):
                self._unirep_singer_data = np.load(self.singer_feature_dir)
                self._unirep_singer_index = np.load(
                    DATA_DIR / "singer_unirep_index.npy"
                )
            index = self._unirep_singer_index[item.name]
            rep = np.concatenate(
                [
                    self._unirep_singer_data[f"{index}_h_avg"],
                    self._unirep_singer_data[f"{index}_h_final"],
                ],
                -1,
            )
            return torch.from_numpy(rep).float()
        elif self.feature_type == FeatureType.ESM1B:
            if not hasattr(self, "_esm_singer_data"):
                from lmdb_dataset import LMDBDataset

                self._esm_singer_data = LMDBDataset(self.singer_feature_dir)
                # self._esm_singer_data = np.load(self.singer_feature_dir)
                self._esm_singer_index = np.load(DATA_DIR / "singer_unirep_index.npy")
            index = self._esm_singer_index[item.name]
            rep = torch.from_numpy(self._esm_singer_data.get(str(index))).float()
            return rep
        else:
            raise NotImplementedError


class SingerDataset(Dataset):

    DATA_FILE = DATA_DIR / "singer_design_refinement_stable_proteins_dataset.csv"

    def __init__(
        self,
        feature_strategy: FeatureTypeStrategy,
        mode: Mode = Mode.TRAIN,
        seed: int = 0,
    ):
        super().__init__()
        self.maybe_download()
        self.feature_strategy = feature_strategy
        self.data = pd.read_csv(self.DATA_FILE)
        self.data = self.data[~self.data["stabilityscore_cnn_calibrated"].isnull()]
        with numpy_seed(seed):
            index = self.data.index
            permutation = np.random.permutation(index)
            cutoffs = (np.array([0, 0.9, 0.95, 1]) * len(index)).astype(np.int)
            idx = {Mode.TRAIN: 0, Mode.VALID: 1, Mode.TEST: 2}[mode]
            start, end = cutoffs[idx : idx + 2]
            indices = sorted(permutation[start:end])
            self.data = self.data.loc[indices]
        self.vocab = self.feature_strategy.build_vocab()

    def maybe_download(self) -> None:
        if not self.DATA_FILE.exists():
            import boto3

            self.DATA_FILE.parent.mkdir(exist_ok=True)
            s3 = boto3.client("s3")
            s3.download_file(
                "stability-prediction", self.DATA_FILE.name, str(self.DATA_FILE)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = item["sequence"]
        value = item["stabilityscore_cnn_calibrated"]

        output = {
            "singer_features": self.feature_strategy.load_singer_features(item),
            "singer_sequence": sequence,
            "singer_tokens": torch.from_numpy(
                self.vocab.encode(sequence, validate=False)
            ),
            "singer_value": value,
        }
        return output

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = {k: [item[k] for item in batch] for k in batch[0]}
        output["singer_tokens"] = collate_tokens(output["singer_tokens"], self.vocab)
        output["singer_features"] = collate_tensors(output["singer_features"])
        output["singer_value"] = torch.tensor(
            output["singer_value"], dtype=torch.float
        ).unsqueeze(1)
        return output


class StabilityDataset(Dataset):

    DATA_FILE = DATA_DIR / "ts50_all.csv"

    def __init__(
        self,
        split: Union[str, int],
        feature_strategy: FeatureTypeStrategy,
        mode: Mode = Mode.TRAIN,
        seed: int = 0,
        singer_augment: bool = False,
    ):
        super().__init__()
        self.maybe_download()
        self.data = pd.read_csv(self.DATA_FILE)
        self.data = self.data[~self.data["TS50_float"].isnull()]
        self.split = self.get_split(split)
        self.mode = mode
        self.seed = seed
        self.feature_strategy = feature_strategy

        self.data = self.split_data()

        self.vocab = self.feature_strategy.build_vocab()
        self.singer_augment = singer_augment
        if self.singer_augment:
            self.singer_data = SingerDataset(feature_strategy, mode, seed)

    def maybe_download(self) -> None:
        if not self.DATA_FILE.exists():
            import boto3

            self.DATA_FILE.parent.mkdir(exist_ok=True)
            s3 = boto3.client("s3")
            s3.download_file(
                "stability-prediction", "ts50_all.csv", str(self.data_file)
            )

    def get_split(self, split: Union[int, str]) -> str:
        if isinstance(split, int):
            split = sorted(self.data["Group"].unique())[split]
        return split

    def split_data(self) -> pd.DataFrame:
        is_test = self.data["Group"] == self.split
        if self.mode == Mode.TEST:
            return self.data[is_test]
        else:
            with numpy_seed(self.seed):
                index = self.data.index
                permutation = np.random.permutation(index)
                cutoffs = (np.array([0, 0.9, 1]) * len(index)).astype(np.int64)
                start, end = cutoffs[:2] if self.mode == Mode.TRAIN else cutoffs[1:]
                ids = sorted(permutation[start:end])
                return self.data.loc[ids]

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx]

        sequence = item["Sequence"]
        ts50 = item["TS50_float"]
        if ts50 > 70:
            ts50 = 70
        label = item["label"]
        parent = item["Label(s)"]
        esm1b_ppl = item["esm1b_ppl"]
        name = item["Name"]

        output = {
            "sequence": sequence,
            "esm1b_ppl": esm1b_ppl,
            "parent": parent,
            "label": label,
            "ts50": ts50,
            "name": name,
        }

        output["tokens"] = torch.from_numpy(self.vocab.encode(sequence, validate=False))
        output["features"] = self.feature_strategy.load_features(item)

        if self.singer_augment:
            index = random.randint(0, len(self.singer_data) - 1)
            output.update(self.singer_data[index])
        return output

    def __len__(self) -> int:
        return len(self.data)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = {k: [item[k] for item in batch] for k in batch[0]}
        output["tokens"] = collate_tokens(output["tokens"], self.vocab)
        if "singer_tokens" in output:
            output["singer_tokens"] = collate_tokens(
                output["singer_tokens"], self.vocab
            )
        if "features" in output:
            output["features"] = collate_tensors(output["features"])
        if "singer_features" in output:
            output["singer_features"] = collate_tensors(output["singer_features"])
        output["esm1b_ppl"] = torch.tensor(output["esm1b_ppl"], dtype=torch.float)
        output["label"] = torch.tensor(output["label"], dtype=torch.long)
        output["ts50"] = torch.tensor(output["ts50"], dtype=torch.float).unsqueeze(1)
        if "singer_value" in output:
            output["singer_value"] = torch.tensor(
                output["singer_value"], dtype=torch.float
            ).unsqueeze(1)
        return output

    @property
    def max_length(self) -> int:
        if not hasattr(self, "_max_length"):
            self._max_length = int(self.data["Sequence"].str.len().max())
        return self._max_length


class BioregDataset(Dataset):
    def __init__(
        self,
        feature_strategy: FeatureTypeStrategy,
        split: str = "random",
        mode: Mode = Mode.TRAIN,
        seed: int = 0,
        singer_augment: bool = False,
    ):
        super().__init__()
        self.data_file = DATA_DIR / "ALL_bioreg_scFvs_iso_cepa.csv"
        self.maybe_download()
        self.data = pd.read_csv(self.data_file)
        self.data = self.data[
            ~(self.data["cepa_Tm1"].isnull() & self.data["iso_Tm1"].isnull())
        ]
        self.mode = mode
        self.seed = seed
        self.feature_strategy = feature_strategy

        self.data = self.split_data()

        self.vocab = self.feature_strategy.build_vocab()
        self.singer_augment = singer_augment
        self.bins = np.arange(35, 80, 10)
        if self.singer_augment:
            self.singer_data = SingerDataset(feature_strategy, mode, seed)

    def maybe_download(self) -> None:
        if not self.data_file.exists():
            import boto3

            self.data_file.parent.mkdir(exist_ok=True)
            s3 = boto3.client("s3")
            s3.download_file(
                "stability-prediction", self.data_file.name, str(self.data_file)
            )

    def split_data(self) -> pd.DataFrame:
        with numpy_seed(self.seed):
            index = self.data.index
            permutation = np.random.permutation(index)
            cutoffs = (np.array([0, 0.9, 0.95, 1]) * len(index)).astype(np.int64)
            start, end = cutoffs[
                {
                    Mode.TRAIN: slice(0, 2),
                    Mode.VALID: slice(1, 3),
                    Mode.TEST: slice(2, 4),
                }[self.mode]
            ]
            ids = sorted(permutation[start:end])
            return self.data.loc[ids]

    def bin_value(self, value):
        return -1 if np.isnan(value) else np.digitize(value, self.bins)

    def __getitem__(self, idx: int):
        item = self.data.iloc[idx]

        sequence = item["PROTSEQFULL_CHAIN"]
        cepa_tm = item["cepa_Tm1"]
        iso_tm = item["iso_Tm1"]
        cepa_label = self.bin_value(cepa_tm)
        iso_label = self.bin_value(iso_tm)
        name = item["ID"]

        output = {
            "sequence": sequence,
            "cepa_label": cepa_label,
            "iso_label": iso_label,
            "cepa_tm": cepa_tm,
            "iso_tm": iso_tm,
            "name": name,
        }

        output["tokens"] = torch.from_numpy(self.vocab.encode(sequence, validate=False))
        output["features"] = self.feature_strategy.load_features(item)

        if self.singer_augment:
            index = random.randint(0, len(self.singer_data) - 1)
            output.update(self.singer_data[index])
        return output

    def __len__(self) -> int:
        return len(self.data)

    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = {k: [item[k] for item in batch] for k in batch[0]}
        output["tokens"] = collate_tokens(output["tokens"], self.vocab)
        if "singer_tokens" in output:
            output["singer_tokens"] = collate_tokens(
                output["singer_tokens"], self.vocab
            )
        if "features" in output:
            output["features"] = collate_tensors(output["features"])
        if "singer_features" in output:
            output["singer_features"] = collate_tensors(output["singer_features"])
        output["cepa_label"] = torch.tensor(output["cepa_label"], dtype=torch.long)
        output["iso_label"] = torch.tensor(output["iso_label"], dtype=torch.long)
        output["cepa_tm"] = torch.tensor(output["cepa_tm"], dtype=torch.float).unsqueeze(1)
        output["iso_tm"] = torch.tensor(output["iso_tm"], dtype=torch.float).unsqueeze(1)
        if "singer_value" in output:
            output["singer_value"] = torch.tensor(
                output["singer_value"], dtype=torch.float
            ).unsqueeze(1)
        return output

    @property
    def max_length(self) -> int:
        if not hasattr(self, "_max_length"):
            self._max_length = int(self.sequences.str.len().max())
        return self._max_length

    @property
    def sequences(self) -> pd.Series:
        return self.data["PROTSEQFULL_CHAIN"]


class PairedStabilityDataset(StabilityDataset):
    def __init__(
        self,
        split: Union[str, int],
        feature_strategy: FeatureTypeStrategy,
        mode: Mode = Mode.TRAIN,
        seed: int = 0,
        randomize: bool = True,
    ):
        super().__init__(split, mode, seed, feature_strategy)
        self.randomize = randomize

    def convert_linear_index(self, idx: int) -> Tuple[int, int]:
        n = len(self.data)
        i = n - 2 - math.floor(math.sqrt(-8 * idx + 4 * n * (n - 1) - 7) / 2.0 - 0.5)
        j = idx + i + 1 - n * (n - 1) // 2 + (n - i) * ((n - i) - 1) // 2
        return i, j

    def __getitem__(self, idx: int):
        idx1, idx2 = self.convert_linear_index(idx)
        item1 = super().__getitem__(idx1)
        item2 = super().__getitem__(idx2)

        if self.randomize and random.random() < 0.5:
            item1, item2 = item2, item1

        return item1, item2

    def __len__(self) -> int:
        return len(self.data) * (len(self.data) - 1) // 2

    def collater(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
        out1 = super().collater([el[0] for el in batch])
        out2 = super().collater([el[1] for el in batch])
        diff = out1["ts50"] - out2["ts50"]
        bins = [-25, -15, -5, 5, 15, 25]
        diff_bins = torch.from_numpy(np.digitize(diff, bins)).squeeze(1)
        return out1, out2, diff_bins
