# Mixed Normalizing Flows
###### Authors: Mathias Niemann Tygesen

This repo contains code to do conditional density estimation with conditional
 normalizing flows with maximum likelihood training.
 
 The code for the flows and training is in the python module DEwNF (Density Estimation with Normalizing Flows).
 The code for the normalizing flows is based on code from _Pyro_ (http://pyro.ai/)
 with training using _PyTorch_ (https://pytorch.org/)
 
 The flows has been trained in two ways:
 - On a GPU enabled compute-cluster via bash and python scripts. 
 For examples see `bash_scripts` and `python_scripts`.
 
 - In GPU enabled Google Colab notebooks. These notebooks are similar to the python scripts but do not need to be 
 executed via bash scripts. For examples see `google_colab_notebooks`.
 
 Code for generating the synthetic data can be found in `DEwNF/samplers` and a data set of
 100,000 samples can be found in `two_moons_data`. Flows trained on this data can be found in `two_moons_models`
 and a plotting Google Colab notebook can be found in `google_colab_notebooks/synthetic_experiments/paper_two_moons_plot`
 
 A preprocessed NYC Yellow Taxi data set can be found in `nyc_data`. The preprocessing is based on https://github.com/hughsalimbeni/bayesian_benchmarks.
 