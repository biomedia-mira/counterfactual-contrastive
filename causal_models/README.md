# Code for counterfactual image generation

The code in this folder is adapted from the official code associated with the  
'High Fidelity Image Counterfactuals with Probabilistic Causal Models' paper. Original code: [https://github.com/biomedia-mira/causal-gen](https://github.com/biomedia-mira/causal-gen).

## Train the counterfactual inference model

To train the counterfactual inference models in the paper you can simply run
`python causal_models/main.py --hps embed --exp_name your_exp_name` replace by `padchest` if you want to train on chest x-rays. All associated hyperparameters are stored in `causal_models/hps.py`. 

This assumes you have already set up your data folders as per the main repository.

## Generating and saving counterfactuals for contrastive training
To generate all possible domain counterfactuals given a trained model, you can use our predefined scripts: `save_embed_scanner_cf.py` and`save_padchest_scanner_cf.py`. Simply pass your checkpoint path and your target saving directory as command line arguments.
