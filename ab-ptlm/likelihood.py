
import os
import urllib
from time import time
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F


import igfold
from igfold import IgFoldInput
from igfold.model.IgFold import IgFold
from igfold.utils.folding import fold, get_sequence_dict, process_template
from igfold.utils.embed import embed
from igfold.utils.general import exists


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


def display_license():
    license_url = "https://github.com/Graylab/IgFold/blob/main/LICENSE.md"
    license_message = f"""
    The code, data, and weights for this work are made available for non-commercial use 
    (including at commercial entities) under the terms of the JHU Academic Software License 
    Agreement. For commercial inquiries, please contact jruffolo[at]jhu.edu.
    License: {license_url}
    """
    print(license_message)


class IgFoldRunner():
    """
    Wrapper for IgFold model predictions.
    """
    def __init__(self, num_models=4, model_idx = 0, model_ckpts=None, try_gpu=True):
        """
        Initialize IgFoldRunner.
        :param num_models: Number of pre-trained IgFold models to use for prediction.
        :param model_ckpts: List of model checkpoints to use (instead of pre-trained).
        """

        display_license()

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
        print( "Model type: ", type(igfold_model) )
        self.device = device
        self.tokenizer = igfold_model.tokenizer
        print( "Tokenizer type: ", type(self.tokenizer) )
        self.bert_model = igfold_model.bert_model
        print( "BERT Model type: ", type(self.bert_model) )

        
    def get_tokens(
        self,
        seq,
    ):  
        print( self.tokenizer )
        print( type(self.tokenizer) )
        print(seq)
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

        print(tokens)

        return tokens.to(self.device)

    
    def likelihood( 
        self, 
        model_idx = 0,
        sequences= None, 
        fasta_file = None,
        template_pdb = None,
        ignore_cdrs = None,
        ignore_chain = None,
    ):
        """
        Obtain the zero-shot likelihoods from AntiBERTy

        :params sequences: Dictionary of sequences
        """
       

        seq_dict = get_sequence_dict( sequences, fasta_file )

        temp_coords, temp_mask = process_template(
            template_pdb,
            fasta_file,
            ignore_cdrs=ignore_cdrs,
            ignore_chain=ignore_chain,
        )

        model_in = IgFoldInput(
            sequences=seq_dict.values(),
            template_coords=temp_coords,
            template_mask=temp_mask,
            return_embeddings=True,
        )


        tokens = [self.get_tokens(s) for s in model_in.sequences]

        def get_likelihoods( tokens ):
        
            bert_output = self.bert_model( 
                tokens, 
                output_hidden_states = True, 
                output_attentions = True 
            )

            print("Object Name : ", type(bert_output))
            print("Object attrs : ", bert_output.__dict__.keys())

            features = bert_output.hidden_states

            #print(len(features))
            #print(features)
            return features


        for t in tokens:
            #print(t)
            f = get_likelihoods( t )

        return True


def main():

    igfoldApp = IgFoldRunner()

    sequences = {
        "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
        "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
    }

    hidden = igfoldApp.likelihood(model_idx=0, sequences=sequences)

    #print(hidden)


if __name__ == '__main__':
    main()


        










