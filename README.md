# CF-SimCLR: counterfactual contrastive learning

This repository contains the code for the paper "Counterfactual contrastive learning: domain-aligned features for improved robustness to acquisition shift".

![alt text](figure1.png)

## Overview
The repository is divided in three main parts:
* The [causal_model/](causal_models/) folder contains all code related to counterfactual inference model training. It contains its own README, giving you all necessary commands to train a DSCM on EMBED and PadChest.
* The [classification/](classification/) folder contains all the code related to self-supervised training as well as finetuning for evaluation (see below).
* The [data_handling/](data_handling/) folder contains everything you need to define your dataset classes. In particular, it contains all the boilerplate for CF-SimCLR specific data loading.
* The [evaluation/](evaluation/) folder contains all the code related to test inference and results plotting for reproducing the plots from the paper. 


## Prerequisites

### Code dependencies
The code is written in PyTorch, with PyTorch Lightning. 
You can install all our dependencies using our conda enviromnent requirements file `environment_gpu.yml'. 

### Datasets
You will need to download the relevant datasets to run our code. 
You can find the datasets at XXX, XXXX, XXX.
Once you have downloaded the datasets, please update the corresponding paths at the top of the `mammo.py` and `xray.py` files.
Additionally, for EMBED you will need to preprocess the original dataframes with our script `data_handling/csv_generation_code/generate_embed_csv.ipynb`. Similarly for RSNA please run first `data_handling/csv_generation_code/rsna_generate_full_csv.py`.


## Full workflow example for training and evaluating CF-SimCLR
Here we'll run through an example to train and evaluate CF-SimCLR on EMBED

1. Train a counterfactual image generation model with 
```
python causal_models/main.py --hps embed
```

2. Generate and save all counterfactuals from every image in the training set with
```
python causal_models/save_embed_scanner_cf.py
```

3. Train the CF-SimCLR model
``` 
python classification/train.py experiment=simclr_embed data.use_counterfactuals=True counterfactual_contrastive=True
```
Alternatively to train a SimCLR baseline just run
``` 
python classification/train.py experiment=simclr_embed
```
Or to run the baseline with counterfactuals added to the training set without counterfactual contrastive objective
``` 
python classification/train.py experiment=simclr_embed data.use_counterfactuals=True counterfactual_contrastive=False
```

4. Train classifier with linear finetuning or finetuning
```
python classification/train.py experiment=base_density trainer.finetune_path=PATH_TO_ENCODER seed=33 trainer.freeze_encoder=True
```
You can choose the proportion of labelled data to use for finetuning with the flag `data.prop_train=1.0`

5. Evaluate on the test set by running the notebook `evaluation/embed_density.ipynb` to run and save inference results on the test set. 
