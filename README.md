# kNNSpectralRates
The code in this repository reproduces the experimental results on convergence rates for k-nearest neighbor graph Laplacians from our paper 

Calder and Garcia Trillos. [Improved spectral convergence rates for graph Laplacians on epsilon-graphs and k-NN graphs](https://arxiv.org/abs/1910.13476). arXiv:1910.13476, 2020.

The scripts rely on the Python package [GraphLearning](https://github.com/jwcalder/GraphLearning), which is required to run the experiments. The results of the experiments are saved in the file `error.csv`. To run the experiments again, run the script
```
python sphere_rates.py > error.csv
```
To generate the plots run
```
python plots.py
```
