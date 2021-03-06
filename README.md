# The Surprising Power of Graph Neural Networks with Random Node Initialization

This repository provides the EXP and CEXP datasets, as well as instructions for reproducing the results of the IJCAI 2021
[paper](https://arxiv.org/pdf/2010.01179.pdf) "The Surprising Power of Graph Neural Networks with Random Node Initialization".

## Datasets
The EXP and CEXP datasets are available in the ''Data''' directory of this repository, both in PyTorch
geometric format (saved as raw pickle), and in .txt format, suitable for the official [PPGN](https://github.com/hadarser/ProvablyPowerfulGraphNetworks) implementation.

### Generating datasets
This repository also contains the file ```GraphGen.py```, which can be used to produce new versions of EXP/CEXP with variable settings.
To run this generator, please set up the following:
- A Python environment (3.6+)
- NetworkX 2.1+
- [Plantri](http://users.cecs.anu.edu.au/~bdm/plantri/) 5.0 (for generating the planar isolated graphs), installed in the same directory
- [PyMiniSolvers](https://pyminisolvers.readthedocs.io/) (for verifying the satisfiability of generated formulas)
- NumPy 1.18.1
- tqdm 4.42
- Matplotlib 2.2.3 (for producing core pair drawings)
- PyTorch 1.4.0 and later 
- PyTorch Geometric (for saving the data in torch.data format)

## Running GNN-RNI
To run GNN-RNI, first install the k-gnn [project](https://github.com/chrsmrrs/k-gnn/). Then, call the file ```GNNHyb.py``` within the used directory.

GNNHyb includes a command-line interface which can be used to set different GNN parameters, such as learning rate,
randomization percentage, dataset choice, and number of GNN layers. To get the full list of options, please consult the help manual via the command:

```python GNNHyb.py -h```

For instance, to run a GNN with 50% randomisation, 3 layers, and uniform RNI on CEXP use the command:

```python GNNHyb.py -dataset CEXP -layers 3 -randomRatio 0.5 -probDist u```

## Running 3-GNN
To use the 3-GNN implementation, found in ```3-GNN.py```, you must first replace ```cpu/isomorphism.h``` within k-gnn with the same file found in ```3-GNN Setup```, and set up the project again. 
The 3-GNN implementation analogously can be called via the command line. These arguments can also be found using the command: 

```python 3-GNN.py -h```

For example, running a 64-dimensional 3-GNN with 2 layers can be run with the command: 

```python 3-GNN.py -layers 2 -width 64```

**NOTE:** When jointly running 3-GNN and GNN-RNI, make sure to use **distinct copies** of the datasets, as the GNN-RNI pre-transform is incompatible with 3-GNN.
## Citing this paper
If you use this code, or its accompanying [paper](https://arxiv.org/pdf/2010.01179), please cite this work as follows:

```
@inproceedings{ACGL-IJCAI21,
  title={The Surprising Power of Graph Neural Networks with Random Node Initialization},
  author    = {Ralph Abboud and {\.I}smail {\.I}lkan Ceylan and Martin Grohe 
               and Thomas Lukasiewicz},
  booktitle={Proceedings of the Thirtieth International Joint Conference on Artifical Intelligence ({IJCAI})},
  year={2021}
}
```

## Acknowledgments
We would like to thank Christopher Morris for his support in setting up k-gnn, and in extending the codebase to support the full 3-GNN.

