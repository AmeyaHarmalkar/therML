# TherML

The official repository for thermostability prediction of single fragment variable chain antibodies (scFvs) and monoclonal antibodies (mAbs) using deep learning. 

The code from this work is made available here. The google colab to try this out will be out soon!
The pre-trained weights will be released for commercial and non-commercial use soon.


### Requirements

We have trained both unsupervised pre-trained language models (PTLMs) and supervised convolutional neural networks (SCNNs) in this work. The PTLMs are in the `./ptlm` directory and the SCNNs are in the `./thermD` directory respectively. The antibody-specific pre-trained language model (trained with AntiBERTy) is in the `./ab-ptlm` directory.
Further details on how to use each of these models is described in the subsequent README files. 

For the PTLMs, the following dependencies are required:

- `pytorch`
- `pytorch-lightning`
- `torchmetrics`
- `esm`
- `evo`

For the SCNNs, the following dependencies are required:

- `pytorch`
- `DeepAb` structure prediction software available via RosettaCommons-DL (This can be updated with `IgFold` for better structural accuracy)
- `torchmetrics`

### Overview

TherML is a proof-of-concept study for general pre-trained language models (UniRep, ESM), antibody-specific language models (AntiBERTy), and supervised energy-based models (thermD). The three directories have code for the three different models being evaluated in this work.
1. `ab-ptlm` : AntiBERTy model (zero-shot and fine-tuned)
2. `ptlm` : General pre-trained language models (zero-shot and fine-tuned)
3. `thermD` : Supervised convolutional networks trained with energies and sequence information
Each of the directories has further information about respective models.

### Bug reports

If you run into any problems while using therML, please create a Github issue with a description of the problem and the steps to reproduce it. You can also reach out to us at `ameya@jhu.edu` and let us know if we can improve our code/help you troubleshoot the bug/s.


### Citing this work

```@article{Harmalkar2023,
author = {Harmalkar, Ameya and Rao, Roshan and Honer, Jonas and Deisting, Wibke and Anlahr, Jonas and Hoenig, Anja and Czwikla, Julia and Rau, Doris and Rice, Austin and Riley, Timothy P and Li, Danqing and Catterall, Hannah B and Tinberg, Christine E and Gray, Jeffrey J and Wei, Kathy Y and Science, Computer and Discovery, Therapeutic and Oaks, Thousand and Francisco, South San},
title = {{Towards generalizable prediction of antibody thermostability using machine learning on sequence and structure features}},
journal = {KMAB-mAbs},
doi = {10.1080/19420862.2022.2163584}
year = {2023}
}
```
