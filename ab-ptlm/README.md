# scFv Stability Prediction with Machine Learning

### Data

The TS50 dataset is largely used for training. Other datasets were collected for evaluation. The training sequence data is the intellectual property of Amgen Research and will not be released in this repository.  


### Requirements

Before running, several dependencies need to be installed:

- `pytorch`
- `pytorch-lightning`
- `torchmetrics`
- `antiberty`
- `igfold`
- `evo`

Most of these are common requirements, however for antibody-specific embeddings we will require antiberty and igfold. Antiberty is the BERT model trained on antibody sequences from Observed Antibody Space. Alternative antibody-specific models are AbLang, AntiBERTa, Progen-OAS. AntiBERTy has been trained on both heavy and light chain sequences. Please refer to [https://github.com/Graylab/IgFold](github.com/Graylab/IgFold) for installation instructions and license information. The models presented in this section were trained on AntiBERTy [https://pypi.org/project/antiberty/](pypi.org/project/antiberty/) for non-commercial use. 

### Training

Code for training the model is located in `./supervised.py`, and can be run via `python supervised.py` There are a number of arguments available, the most important of which are listed here.

- `split`: Which dataset split of TS50 to run on. Options are integers in [0, 17), which correspond to the alphabetical order of the selected splits. The string "all" can also be passed in (and is the default), which will run all splits. 
- `base_model`: Which base model features to use. Choices are limited to "antiberty" for now. See note on feature saving below.
- `head_model`: Which head model to use. Choices are "attnmean" and "concat", default is "concat". This only has an effect for some models.
- `mlp`: Whether or not to use a nonlinear classifier on top of the extracted features.

### Features

The antibody-specific PTLM model has been used for generating zero-shot predictions and has been fine-tuned on all experimental sets (n-1 sets used for training).
The zero-shot models can be used directly on sequences. More details for that are provided in  `./utility/get_antibody_likelihoods.py`.
The fine-tuned models are trained and stored in the `./logs` directory. Code for loading these features is in `./dataset.py`, in the class `FeatureTypeStrategy`.




