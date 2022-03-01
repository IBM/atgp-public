# AT-GP in Python 
A python implementation of the apdative transfer learning algorithm for Gaussian processes (AT-GP) proposed by Cao et al. in 2010 (Bin Cao, Sinno Jialin Pan, Yu Zhang, Dit-Yan Yeung, Qiang Yang: Adaptive Transfer Learning, in Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence (AAAI) 2010).
The repository also contains  code to evaluate the prediction accuracy of AT-GP using the Styblinski-Tang function (<https://www.sfu.ca/~ssurjano/stybtang.html>)  and the Goldstein-Price function (<https://www.sfu.ca/~ssurjano/goldpr.html>). 

The main folder of the repository is AT-GP and it contains the following sub-folders:

## ./models
Contains the implementation of the ATGP kernel. 

## ./functions
Contains the different implementations of the synthetic functions used for the experiments.

## ./data
Contains data generated from the synthetic functions and used for the experiments.

## ./plots
Contains scripts to plot the result of the different experiments and to plot the synthetic functions.

# License

This project is licensed under the Apache License 2.0.
If you would like to see the detailed LICENSE click [here](LICENSE).

# Contributing

Please see [CONTRIBUTING](CONTRIBUTING.md) for details.
Note that this repository has been configured with the [DCO bot](https://github.com/probot/dco).
