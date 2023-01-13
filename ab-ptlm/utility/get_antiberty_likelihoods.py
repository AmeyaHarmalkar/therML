
import os
import urllib
from time import time
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
from itertools import product, zip_longest


import igfold
from igfold import IgFoldInput
from igfold.model.IgFold import IgFold
from igfold.utils.folding import fold, get_sequence_dict, process_template
from igfold.utils.embed import embed
from igfold.utils.general import exists

from antiberty import AntiBERTy, get_weights

from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import BertTokenizer

def download_ckpts():
    """
    Download checkpoint files if not found.
    """
    print("Downloading checkpoint files...")

    tar_file = "IgFold.tar.gz"
    ckpt_url = f"https://data.graylab.jhu.edu/{tar_file}"

    project_path = os.path.dirname(os.path.realpath(igfold.__file__))
    ckpt_dir = os.path.join(
        project_path,
        "trained_models/",
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_tar_file = os.path.join(project_path, tar_file)

    urllib.request.urlretrieve(ckpt_url, ckpt_tar_file)
    os.system(f"tar -xzf {ckpt_tar_file} -C {ckpt_dir}")
    os.remove(ckpt_tar_file)


class AntiBERTyRunner():
    """
    Wrapper for IgFold model predictions.
    """
    def __init__(self, num_models=4, model_idx = 0, model_ckpts=None, try_gpu=True):
        """
        Initialize IgFoldRunner.
        :param num_models: Number of pre-trained IgFold models to use for prediction.
        :param model_ckpts: List of model checkpoints to use (instead of pre-trained).
        """

        if exists(model_ckpts):
            num_models = len(model_ckpts)
        else:
            if num_models < 1 or num_models > 4:
                raise ValueError("num_models must be between 1 and 4.")

            if not exists(model_ckpts):
                project_path = os.path.dirname(
                    os.path.realpath(igfold.__file__))

                ckpt_path = os.path.join(
                    project_path,
                    "trained_models/IgFold/*.ckpt",
                )
                if len(glob(ckpt_path)) < num_models:
                    download_ckpts()
                model_ckpts = list(glob(ckpt_path))

            model_ckpts = list(sorted(model_ckpts))[:num_models]

        print(f"Loading {num_models} IgFold models...")

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and try_gpu else "cpu")
        print(f"Using device: {device}")

        self.models = []
        for ckpt_file in model_ckpts:
            print(f"Loading {ckpt_file}...")
            self.models.append(
                IgFold.load_from_checkpoint(ckpt_file).eval().to(device))


        print(f"Successfully loaded {num_models} IgFold models.")

        igfold_model = self.models[model_idx]
        self.device = device
        self.tokenizer = igfold_model.tokenizer
        self.bert_model = igfold_model.bert_model
        self.antiberty = AntiBERTy.from_pretrained(get_weights()).to(device)
        self.antiberty.requires_grad = False
        self.aby_heads = self.antiberty.cls
        self.vocab_size = len(self.tokenizer.vocab)
        self.vocab = self.tokenizer.vocab

        
    def get_tokens(
        self,
        seq,
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

    
    def likelihoods( 
        self,
        sequences= None, 
        fasta_file = None,
        template_pdb = None,
        ignore_cdrs = None,
        ignore_chain = None,
    ):
        """
        OObtains the likelihoods from prediction-logits
        """

        tokens = [self.get_tokens(s) for s in sequences.values()]
        seq_values = [s for s in sequences.values()]

        logits = []
        labels = []

        for t in tokens:
            berty_output = self.antiberty(t)
            predictions = berty_output.prediction_logits
            #predictions = F.softmax(berty_output.prediction_logits, -1)
            logits.append( torch.squeeze(predictions, 0) )
            labels.append( torch.squeeze(t, 0) )
        
        labels = torch.cat(labels, -1)
        logits = torch.cat(logits)
        
        pseudo_ppl = nn.CrossEntropyLoss( reduction="mean" )( logits, labels)

        return logits, pseudo_ppl.item()


    def masked_logits(
        self,
        sequence,
    ):
        """
        With a mask across each residue, estimate the prediction-logits,
        compares with the token and calculates a zero-shot likelihood.
        """

        logits = []
        labels = []
        
        tokens = [self.get_tokens(s) for s in sequence.values()]
        for i in range(len(tokens)):
            token = torch.unsqueeze( tokens[i].repeat( len(tokens[i][-1] ), 1 ).fill_diagonal_(4), 1)
            labels.append( torch.squeeze( tokens[i], 0 ) )
            for j,t in enumerate(token):
                berty_output = self.antiberty(t)
                predictions = berty_output.prediction_logits
                logits.append( predictions[0][j] )
        
        logits = torch.stack((logits))
        labels = torch.cat( labels, -1 )

        pseudo_ppl = nn.CrossEntropyLoss( reduction="mean" )( logits, labels)
        return logits, pseudo_ppl.item()


    
def sequence_pseudo_ppl( logits, tokens ):

    pseudo_ppl = nn.CrossEntropyLoss( reduction="mean" )( logits, tokens )
    return pseudo_ppl.exp().item()


def main():

    igfoldApp = AntiBERTyRunner()

    df = pd.read_csv( 'data/all_ts50_data.csv' )
    df_sequences = df["Sequence"]
    df_group = [x for x in df["Group"]]
    df_name = [x for x in df["Name"]]
    df_ts50 = [x for x in df["TS50_float"]]

    linkers_list = [
                "GGGGS" * 3,  # G4Sx3
                "GGGGS" * 2,  # G4Sx2
                "GGGGSGGGSGGGGS",  # G4SG3SG4S - likely mistranscription
                "GGGGSGGGGPGGGGS",  # G4SG4PG4S - likely mistranscription
            ]

    def split_linkers(seq):
        for linker in linkers_list:
            try:
                heavy, light = seq.split(linker)
            except ValueError:
                continue
            
            return heavy, light
        else:
            raise RuntimeError(f"No appropriate linker found: {seq}")

    sequences = []

    for seq in df_sequences:
        heavy, light = split_linkers(seq)
        sequences.append( {'H': heavy, 'L': light }  )

    net_likelihood = [ ]

    for i in range(len(sequences)):
        _, likelihoods = igfoldApp.masked_logits(sequences[i])
        #_, likelihoods = igfoldApp.likelihoods(sequences[i])
        net_likelihood.append( [ df_group[i], df_name[i], df_ts50[i], likelihoods ] )

    data_list = list(map( list, zip(*net_likelihood) ))    
    df = pd.DataFrame(np.transpose(data_list))
    df.columns = [ 'Group', 'Name', 'TS50', 'Likelihoods' ]
    print(df)
    df.to_csv( 'AntiBERTy_zeroshot_updated_llh' +'.csv', index=False )


if __name__ == '__main__':
    main()

