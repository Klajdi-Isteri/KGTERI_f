# KG-TERI

This is the official repo of the dissertation
[*Exploiting Time and Content Information to Improve Collaborative and Knowledge-aware Recommendation*](https://drive.google.com/uc?id=1FYUNfTY7QPwGlgm3tL7M5a1LqHjay_eQ&export=download) which involves the implementation of a new *Graph Neural Network* model based on the approaches discussed in papers submitted/accepted at *SIGIR Conference on Research and Development in Information Retrieval*.

Huge thanks to the [@sisinflab](https://github.com/sisinflab) team for their support.
## Table of Contents

- [Description](#description)
- [Requirements](#requirements)
  - [Installation guidelines: scenario #1](#installation-guidelines-scenario-1)
  - [Installation guidelines: scenario #2](#installation-guidelines-scenario-2)
- [Datasets](#datasets)
- [Elliot Configuration Files](#elliot-configuration-files)
- [Usage](#usage)
  - [Matrices](#matrices)
  - [Forwarding](#forwarding)
  - [Reproduce Results](#reproduce-results)



## Description

The code in this repository allows replicating the experimental setting described within the dissertation.

The recommenders training and evaluation procedures have been developed on the reproducibility framework **Elliot**,
so we suggest you refer to the official GitHub 
[page](https://github.com/sisinflab/elliot) and 
[documentation](https://elliot.readthedocs.io/en/latest/).

Regarding the graph-based recommendation models based on torch, they have been implemented
in `PyTorch Geometric` using the version `1.10.2`, with CUDA `10.2` and cuDNN `8.0`

For granting the usage of the same environment on different machines, 
all the experiments have been executed on the same docker container.
If the reader would like to use it, 
please look at the corresponding section in [requirements](#requirements).

## Requirements 

This software has been executed on the operative system Ubuntu `18.04`.

Please, make sure to have the following installed on your system:

* Python `3.8.0`
* PyTorch Geometric with PyTorch `1.10.2` or later
* CUDA `10.2`

### Installation guidelines: scenario #1
If you have the possibility to install CUDA on your workstation (i.e., `10.2`), you may create the virtual environment with the requirements files we included in the repository, as follows:

```
# PYTORCH ENVIRONMENT (CUDA 10.2, cuDNN 8.0)

$ python3.8 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip install -r requirements.txt
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html
```

### Installation guidelines: scenario #2
A more convenient way of running experiments is to instantiate a docker container having CUDA `10.2` already installed.

Make sure you have Docker and NVIDIA Container Toolkit installed on your machine (you may refer to this [guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)).

Then, you may use the following Docker image to instantiate the container equipped with CUDA `10.2` and cuDNN `8.0` (the environment for `PyTorch`): [link](https://hub.docker.com/layers/nvidia/cuda/10.2-cudnn8-devel-ubuntu18.04/images/sha256-3d1aefa978b106e8cbe50743bba8c4ddadacf13fe3165dd67a35e4d904f3aabe?context=explore)

After the setup of your Docker containers, you may follow the exact same guidelines as [scenario #1](#installation-guidelines-scenario-1).

## Datasets

At `./data/` you may find all the [files](data) related to 
the datasets, the knowledge graphs and the related item-entity linking.

The datasets could be found within the directory `./data/[DATASET]/data`. 
Only for Movielens 1M, within the [directory](data/movielens/grouplens) `./data/movielens/grouplens`
For the knowledge graphs and links please look at  `./data/[DATASET]/dbpedia`.

NOTE: the dataset has been reduced to Movielens 10k with the use of `./external/models/kgteri/Movielens_prep.py`.
Also the knowledge graph side information has been reduced to the number of users obtained from Movielens 10k for a direct and correct mapping.

## Elliot Configuration Files

At `./config_files/` you may find the Elliot [configuration files](config_files) used for setting the experiments.


The configuration files for training the models are reported as `[DATASET]_[MODEL].yml`. 
While the best models hyperparameters are reported in the files named `[DATASET]_best_[MODEL].yml`.

## Usage
Here we describe the steps to reproduce the results presented in the dissertation. Furthermore, we provide a description of how the experiments have been configured.

### Matrices

All the R matrices containing the item-item time and content information are stored in the [directory](data/movielens/kgteri) `./data/movielens/kgteri/user_item_matrix.pk`.
To compute the matrices with different values of k just change it in each external model's `TimeAwareProcessing.py` and uncomment the following lines in the `.external/models/[MODEL]/[MODEL].py`
```
# Run experiment one time to create the pickle file containing the users matrices

self.app = TimeAwareProcessing(data=self._data)
self.app.time_mapping()

```
NOTE: the next experiment run is the one which performs the embeddings forwarding with the calculated matrix of the previous step.

### Forwarding
To perform the approach discussed in the dissertation, just change the type of forwarding in each applied model `.external/models/[MODEL]/[MODEL]Model.py` to the KGTERI Forwarding.



### Reproduce Results

[Here](start_experiments.py) you can find a ready-to-run Python file with all the pre-configured experiments cited in the dissertation.
You can easily run them with the following command:

```
python start_experiments.py
```

NOTE: it runs the experiments with the R matrix obtained with a value of neighbours of ten, for further versions compute the new matrix as in [Matrices](##Matrices).
The results will be stored in the folder ```results/DATASET/```.

