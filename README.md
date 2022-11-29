# Shortest Path Message Passing Neural Networks

This repository contains the source code for the Shortest Path Message Passing Neural Network (SP-MPNN) framework 
models, presented in the LoG 2022 [paper](https://arxiv.org/abs/2206.01003) "Shortest Path Networks for Graph 
Property Prediction". The repository includes the `h-Prox` datasets, as well as all evaluation datasets and code for 
training and testing the different SP-MPNN variants across all experiments in the paper.

## Requirements

The requirements for the Python environment can be found in ``requirements.txt``. The
main packages that have been used are PyTorch, PyTorch Geometric, OGB (Open Graph Benchmark), 
Neptune (neptune.ai) and Numpy.

Running on a GPU is supported via CUDA.

## Datasets

### TU Datasets (Chemical)

Available in Pytorch Geometric.

### Proximity

The synthetic Prox datasets (``k = {1, 3, 5, 8, 10}``) are available [here](https://zenodo.org/record/6557736#.YoPGtS8w30o). 
To use these datasets, the contents of the aforementioned zip file must be extracted and moved into the ``data`` directory

### MoleculeNet

We use the OGB implementation of the MoleculeNet datasets. More information can be found 
[here](https://ogb.stanford.edu/docs/graphprop/). 

### QM9

The QM9 dataset is provided in ``data/QM9``.

## Running

The script we use to run the experiments is ``src/main.py``. Note that the script should be run from inside 
the ``src`` directory, or mark it as Source Root.

The main parameters of the script are:

- ``--dataset`` the dataset we use.
- ``--model`` for the model used. The main convention is ``SP_{INNER}_{OUTER}``, where ``INNER`` corresponds to 
the approach we use for aggregating the embeddings on each hop level, while ``OUTER`` is the approach we use for 
aggregating the different hop levels. The main models we use are ``SP_SUM_WEIGHT``, ``SP_EDGESUM_WEIGHT`` and
``SP_RSUM_WEIGHT``, where the common ``WEIGHT`` outer aggregation is the normalised sum that the simple model in the
paper uses (SPN).
- ``--mode`` for the current task type. We use ``gc`` for Graph Classification, 
and ``gr`` for Graph Regression. 

Additionally, some of the more useful configurable parameters are:

- ``--emb_dim`` for the embedding dimensionality.
- ``--batch_size`` for the batch size we use during training.
- ``--lr`` for the learning rate.
- ``--dropout`` for the dropout probability.
- ``--epochs`` for the number of epochs.

A detailed list of all additional arguments can be seen using the following command:

``python main.py -h``

## Example running 

You can run the end-to-end experiments for the datasets from each class using the commands below, where the 
arguments are replaced with:

- ``{dataset}`` is the dataset we want to run the experiment on 
- ``{k}`` is the maximum distance for a SP layer 
- ``{L}`` is the number of layers
- ``{QM9_TASK}`` is an integer id for the QM9 objective / task we want to predict

### TU Datasets (Chemical), Prox

``python main.py -d {dataset} -m SP_SUM_WEIGHT --max_distance {k} --num_layers {L} --mode gc``

### MoleculeNet

``python main.py -d {dataset} -m SP_EDGESUM_WEIGHT --max_distance {k} --num_layers {L} --mode gc``

### QM9

``python main.py -d QM9 -m SP_RSUM_WEIGHT --max_distance {k} --num_layers {L} --specific_task {QM9_TASK} --mode gr``

## Neptune.ai

You can use [neptune.ai](https://neptune.ai) 
to track the progress, by specifying your project and token in ``src/config.ini``.  Leave the
fields as ``...`` if you want to just run locally.

##  Citing this paper
If you make use of this code, or its accompanying [paper](https://arxiv.org/abs/2206.01003), 
please cite this work as follows:

```
@inproceedings{ADC-LoG2022,
  title={Shortest Path Networks for Graph Property Prediction},
  author    = {Ralph Abboud and Radoslav Dimitrov and
                {\.I}smail {\.I}lkan Ceylan},
  booktitle={Proceedings of the First Learning on Graphs Conference ({LoG})},
  year={2022},
  note={Oral presentation}
}
```
