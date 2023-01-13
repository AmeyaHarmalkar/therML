from typing import List, Union
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from supervised import StabilityModel
from dataset import FeatureType, FeatureTypeStrategy


class SupervisedAntiBERTyScorer:
    "This class scores the predictions and obtains required probabilities"

    def __init__(self):
        self.model = EnsemblePredictorModel()

    def score( self, sequences: List[str] ) -> np.ndarray:
        return np.array( [self.compute_measurements(seq) for seq in tqdm(sequences) ] )
    
    def compute_likelihood(self, seq:str) -> float:
        return self.model.probability(self.model.predict(seq))

    def compute_measurements( self, seq:str ) -> list:
        measurement, probs = self.model.predict(seq)
        return self.model.collate_data( measurement, probs )

    def summary(self):
        print(self.model.head_models[0])
        return self.model.head_models[0]




class EnsemblePredictorModel:
    "This class performs inference on the input sequence"

    def __init__(
        self,
        features: str ="antiberty",
        head: str = "concat",
        mlp: bool = True,
        checkpoint_path: Union[str, Path] = Path(__file__).parents[1] / "logs",
        output_path: Union[str, Path] = Path(__file__).parents[1] / "outputs",
        try_gpu: bool = True,
    ):
        self.features = FeatureType[features.upper()]
        self.head = head
        self.mlp = mlp
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
        self.vocab = self.strategy.build_vocab()
        self.tokenizer = self.base_model.models[0].tokenizer

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and try_gpu else "cpu")
        
        # Bins obtained by averaging out the values of sequences in each of the four bins
        self.bin_values = torch.tensor(
            [36.0395, 56.6874, 64.190, 70.0740], device="cuda:0"
        )
        
        self.output_path.mkdir( exist_ok=True )
        ts50_output_path = self.output_path/ f"ts50_{self.make_name(remove_nonvariable=True)}.csv"
        predictions = pd.read_csv( ts50_output_path )
        self.kde = KernelDensity(kernel="gaussian")
        self.kde.fit(np.asarray(predictions["E[TS50]"])[:, None])


    def make_name(
        self,
        remove_nonvariable: bool = False,
    ):
        name = []
        name.append( f"base{self.features.value.lower()}" )
        name.append( f"head{self.head}" )
        if self.mlp:
            name.append("mlp")
        if not remove_nonvariable:
            name.append("seed0_lr0.001_batch64")
        return "_".join(name)


    def predict(
        self,
        sequence: str,
    ):  
        tokens = [self.get_tokens(s) for s in sequence.values()]
        reps = torch.from_numpy(np.array(self.strategy.forward( self.base_model, [sequence] )))
        predictions = torch.stack(
            [ model(features=reps.to(self.device))["prediction"]for model in self.head_models ], 0
        ).mean(0)

        probabilities = predictions.softmax(-1)
        expected_value = probabilities @ self.bin_values
        return expected_value.item(), torch.squeeze(probabilities, 0)


    def probability(self, prediction: float) -> float:
        return np.exp(self.kde.score(np.array([[prediction]])))

    def collate_data( self, measurement:float, probs:list ) -> list:
        data_list = []
        data_list.append(measurement)
        for ele in probs:
            data_list.append( ele.item() )
        data_list.append( np.argmax(probs.cpu().detach().numpy()).item() )
        return data_list



    def get_tokens(
        self,
        seq: str,
    ):  

        if isinstance(seq, str):
            tokens = self.tokenizer.encode(
                " ".join(list(seq)),
                return_tensors="pt",
            )
        elif isinstance(seq, list) and isinstance(seq[0], str):
            seqs = [" ".join(list(s)) for s in seq]
            tokens = self.tokenizer.batch_encode_plus(
                seqs,
                return_tensors="pt",
            )["input_ids"]
        else:
            tokens = seq

        return tokens.to(self.device)
    
    
    






    


