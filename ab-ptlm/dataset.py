from typing import Union, Tuple, Dict, List, Any
import tarfile
import enum
import hashlib
import contextlib
import math
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from filelock import FileLock
import esm
from evo.tokenization import Vocab
from evo.tensor import numpy_seed, collate_tensors
# To obtain AntiBERTy embeddings
import igfold
# To get AbLANG embeddings
#import ablang


DATA_DIR = Path(__file__).absolute().parent.with_name("data")
AB_DIR = Path(__file__).absolute().parent.with_name("ab-ptlm")
SCRATCH = Path(__file__).absolute().parent.parent.with_name("scr16_jgray21")

def collate_tokens(tokens, vocab: Vocab):
    if tokens[0].dtype == torch.long:
        tokens = collate_tensors(tokens, constant_value=vocab.pad_idx)
    else:
        tokens = collate_tensors(tokens, constant_value=0)
    return tokens


def split_linkers(seq : str) :
    """Function to split the scFv sequence into heavy and light"""

    linkers_list = [
            "GGGGS" * 3,  # G4Sx3
            "GGGGS" * 2,  # G4Sx2
            "GGGGSGGGSGGGGS",  # G4SG3SG4S - likely mistranscription
            "GGGGSGGGGPGGGGS",  # G4SG4PG4S - likely mistranscription
        ]

    for linker in linkers_list:
        try:
            heavy, light = seq.split(linker)
        except ValueError:
            continue    
        return heavy, light
    else:
        raise RuntimeError(f"No appropriate linker found: {seq}")



class FeatureType( enum.Enum ):
    """Define the type of model to load
    """
    ESM1B = "ESM1B"
    ANTIBERTY = "ANTIBERTY"
    ABLANG = "ABLANG"

    @property
    def is_antiberty(self) -> bool:
        return self == FeatureType.ANTIBERTY
    
    @property
    def is_ablang(self) -> bool:
        return self == FeatureType.ABLANG

    @property
    def is_oas(self) -> bool:
        return self.is_antiberty or self.is_ablang

    @property
    def is_single_vector(self) -> bool:
        #return self.is_antiberty
        return False


class Mode(enum.Enum):
    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"


class FeatureTypeStrategy:

    _BUILT_MODEL = None

    FEATURE_DIRS = {
        FeatureType.ESM1B: "esm1b",
        FeatureType.ANTIBERTY: "antiberty",
        FeatureType.ABLANG: "ablang",
    }

    def __init__(self, feature_type: FeatureType, dataset: str = "ts50"):
        self.dataset = dataset
        self.feature_type = feature_type
        self.feature_dir = DATA_DIR / (f"{dataset}_" + self.FEATURE_DIRS[self.feature_type])


    def build_model(self):
        
        if self._BUILT_MODEL is not None:
            return self._BUILT_MODEL
        if self.feature_type == FeatureType.ESM1B:
            model, _ = esm.pretrained.esm1b_t33_650M_UR50S()
        if self.feature_type == FeatureType.ANTIBERTY:
            model = igfold.IgFoldRunner()
        elif self.feature_type == FeatureType.ABLANG:
            # Note: ablang is trained separately on heavy and light chains
            # We need to incorporate that in our model
            model_heavy = ablang.pretrained("heavy")

            model_heavy = ablang.pretrained("heavy")
            model_heavy.freeze()
            model_light = ablang.pretrained("light")
            model_light.freeze()
            model = [ model_heavy, model_light ]
        else:
            raise NotImplementedError(self.feature_type)

        #model = model.eval().requires_grad_(False)
        #model = [model.models[0]]
        self.__class__._BUILT_MODEL = model
        return model


    def build_vocab(self):

        if self.feature_type == FeatureType.ESM1B:
            alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
            return Vocab.from_esm_alphabet(alphabet)

        if self.feature_type == FeatureType.ANTIBERTY:
            ## Do we need to take in the vocab?
            from antiberty_vocab import AntibertyVocab

            return AntibertyVocab()
        elif self.feature_type == FeatureType.ABLANG:
            ## Do we need to take in the vocab?
            raise NotImplementedError(self.feature_type)
        else:
            raise NotImplementedError(self.feature_type)

    def forward(self, model, tokens):

        if self.feature_type == FeatureType.ESM1B:
            return model(tokens, repr_layers=[33])["representations"][33][:, 1:-1]
        if self.feature_type == FeatureType.ANTIBERTY:
            embedding = []
            for i in range(len(tokens)):
                emb = model.embed( sequences=tokens[i] )
                embedding.append( emb.bert_embs.cpu().detach().numpy() )
                del emb
            
            # Pad the extra length
            n_max = 241
            for i in range(len(embedding)):
                n_max = max( n_max, embedding[i].shape[1] )
            embedding = [np.pad(embedding[i][0],([0,n_max-embedding[i].shape[1]],[0,0])) for i in range(len(embedding))]
            
            return embedding
        elif self.feature_type == FeatureType.ABLANG:
            embedding = []
            for i in range(len(tokens)):
                heavy = tokens[i]['H']
                light = tokens[i]['L']
                emb_H = model[0](heavy, mode='rescoding')
                emb_L = model[1](light, mode='rescoding')
                embedding.append(np.concatenate((emb_H, emb_L),axis=1))
                del heavy, light
            return embedding
        else:
            raise NotImplementedError(self.feature_type)

    
    def output_dim(self) -> int:
        if self.feature_type == FeatureType.ESM1B:
            return 1280
        if self.feature_type == FeatureType.ANTIBERTY:
            return 512
        elif self.feature_type == FeatureType.ABLANG:
            return 512
        else:
            raise NotImplementedError(self.feature_type)


    def item_to_feature_path(self, item: pd.Series) -> Path:
        if self.feature_type == FeatureType.ESM1B:
            if self.dataset == "ts50":
                return SCRATCH / f"aharmal1/base-model-features/ts50_esm1b/ts50_{item.name}.pt"
            else:
                return SCRATCH / f"{item.name}.pt"
        elif self.feature_type == FeatureType.ANTIBERTY:
            if self.dataset == "ts50":
                return SCRATCH / f"aharmal1/base-model-features/ts50_antiberty.pkl"
            else:
                return SCRATCH / f"{item.name}.pkl"
        elif self.feature_type == FeatureType.ABLANG:
            return item.name
        else:
            return self.feature_dir / f"{item['Name']}.pkl"


    def load_features(self, item: pd.Series ) -> torch.Tensor:

        path = self.item_to_feature_path(item)

        if self.feature_type.is_antiberty:
            infile = open( path, 'rb')
            rep = pickle.load(infile)
            infile.close()
            #return torch.FloatTensor(rep[item.name].flatten())
            return torch.FloatTensor(rep[item.name])

        rep = torch.load(path)
        if self.feature_type == FeatureType.ESM1B:
            return rep["representations"][33]
        
    

class StabilityDataset( Dataset ):

    DATA_FILE = DATA_DIR / "ts50_all.csv"

    def __init__(
        self,
        split: Union[str, int],
        feature_strategy: FeatureTypeStrategy,
        mode: Mode = Mode.TRAIN,
        seed: int = 0
    ):
        super().__init__()
        self.data = pd.read_csv( self.DATA_FILE )
        self.data = self.data[~self.data["TS50_float"].isnull()]
        self.split = self.get_split(split)
        self.mode = mode
        self.seed = seed
        self.feature_strategy = feature_strategy
        self.data = self.split_data()
        self.vocab = self.feature_strategy.build_vocab()


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
        #esm1b_ppl = item["esm1b_ppl"]
        name = item["Name"]

        output = {
            "sequence": sequence,
            #"esm1b_ppl": esm1b_ppl,
            "parent": parent,
            "label": label,
            "ts50": ts50,
            "name": name,
        }

        output["tokens"] = torch.from_numpy(self.vocab.encode(sequence, validate=False))
        output["features"] = self.feature_strategy.load_features(item)
        return output


    def __len__(self) -> int:
        return len(self.data)


    def collater(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        output = {k: [item[k] for item in batch] for k in batch[0]}
        output["tokens"] = collate_tokens(output["tokens"], self.vocab)
       
        if "features" in output:
            output["features"] = collate_tensors(output["features"])
        
        #output["esm1b_ppl"] = torch.tensor(output["esm1b_ppl"], dtype=torch.float)
        output["label"] = torch.tensor(output["label"], dtype=torch.long)
        output["ts50"] = torch.tensor(output["ts50"], dtype=torch.float).unsqueeze(1)
        
        return output
    

    @property
    def max_length(self) -> int:
        if not hasattr(self, "_max_length"):
            self._max_length = int(self.data["Sequence"].str.len().max())
        return self._max_length
    


        


