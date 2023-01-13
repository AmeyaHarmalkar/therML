# scFv Stability Prediction with Machine Learning

### Data

The TS50 dataset is largely used for training. Other datasets were collected for evaluation. The training sequence data is the intellectual property of Amgen Research and will not be released in this repository. 

### Requirements

Before running, several dependencies need to be installed:

- `pytorch`
- `pytorch-lightning`
- `torchmetrics`
- `esm`
- `evo`

Most of these are common requirements, with the exception of `evo`, which is a personal repository of common public functions to do things like distribute commands across GPUs, read in alignments, etc. We have included a default copy of this within this repository. It can also be cloned from github at [https://github.com/rmrao/evo](https://github.com/rmrao/evo). If you run `cd evo; pip install -e .`, it will not only install `evo`, it will also install `esm`.

### Training

Code for training the model is located in `./supervised-stability`, and can be run via `python supervised.py` There are a number of arguments available, the most important of which are listed here.

- `split`: Which dataset split of TS50 to run on. Options are integers in [0, 17), which correspond to the alphabetical order of the selected splits. The string "all" can also be passed in (and is the default), which will run all splits. 
- `base_model`: Which base model features to use. Choices are "esm1b", "esmmsa", "unirep", "unirep_evotune_msa", "unirep_evotune_oas", "deepab". See note on feature saving below.
- `head_model`: Which head model to use. Choices are "attnmean" and "concat", default is "concat". This only has an effect for some models (ESM models). Other models come with their own default head.
- `mlp`: Whether or not to use a nonlinear classifier on top of the extracted features.
- `regression`: Predict with a regression based loss instead of a classification based loss
- `singer_augment`: Use Singer et al. 2021 data to augment the training.

### Features

When training a model it is very expensive to run the base model. Since we don't fine-tune the base model, we can instead run the model on all inputs and write the features out to a separate file (located in `./base-model-features`). Code for loading these features is in `supervised-stability/dataset.py`, in the class `FeatureTypeStrategy` with the methods `load_features` and `load_singer_features`.

Some of the base models listed in `supervised.py` may not work - for models which did not seem to work well and where feature storage took a huge amount of space, we ended up deleting the features.

## Sequence Design

For sequence design, we settled on two models. First, the ESM-1v unsupervised moel, which relies only on the public `esm` repo and not on any other code here. Second, the ESM-1b Supervised + Head Concat model. This model is exported in easy-to-use form by class defined in `supervised-stability/inference.py`, `EnsemblePredictionModel`. This provides two methods, `predict` and `probability`, which predict the base model value and an estimate of the probability of that value. The probability is obtained using a KDE estimate trained on the model's predictions for the TS50 dataset.

Using the ESM models for the first time requires them to be downloaded. This happens automatically when they are created but for the ESM-1v models can be quite slow (since there are a number of models). We have included the `download_esm1v.sh` script which should download them automatically in the correct location (you may need to create some folders if they're missing).

Each of the below scripts takes two options for `model`:
- `esm1v_unsupervised`
- `esm1b_supervised`

### Single Mutants

Predicting single mutants can be done with the `predict_single_mutant.py` script, passing in an option for a model and a fasta file with a particular sequence. The script will score all possible single mutants. Note unlike the multi-mutant scripts, this will not screen out undesired positions / cysteine mutations, it provides the full single mutant scan.

### Top Mutant Rescoring

The first, simple method of predicting the best multi-mutants takes the top scoring mutants from the single mutant outputs, then combines them. The result is then re-scored to re-order them, potentially accounting for epistasis. The script to do this is `predict_multi_mutant.py`

Arguments:
- `model`: One of `esm1v_unsupervised` or `esm1b_supervised`
- `fastq`: Path to .fastq file containing headers, sequences, and non-mutatable regions (e.g. CDR/linker) marked with '*'
- `max_mutations`: Maximum total number of mutations (default: 3).
- `single_mutant_data_file`: Data file containing output of `predict_single_mutant.py`.
- `num_mutants_to_score`: Total number of mutant combinations to generate and score (default: 120).

### MCMC

The second method of predicting the best multi-mutants uses MCMC to sample mutants. The base mcmc file is `mcmc.py`, and the script used to actually run mcmc is `run_mcmc.py`. 

Arguments:
- `model`: One of `esm1v_unsupervised` or `esm1b_supervised`
- `fastq`: Path to .fastq file containing headers, sequences, and non-mutatable regions (e.g. CDR/linker) marked with '*'
- `max_mutations`: Maximum total number of mutations (default: 3).
- `num_mcmc_steps`: Number of steps of MCMC to run in each trajectory.
- `num_restarts`: Number of trajectories to run for each sequence.