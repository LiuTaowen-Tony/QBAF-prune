# Iterative Pruning to learn QBAF structure

This is a repo for iterative pruning to learn QBAF structure


## Algorithm

```
procedureITERATIVEPRUNING(D,M,η,p,s)
  while Sparsity of M < s do
    Train M on D using learning rate η with l2 regularisation α
    Compute the absolute value of all weights |wi|
    Sort the weights by |wi|
    Set the lowest p percent of weights to zero
    Set the pruned weight to be not trainable
  end while
end procedure
```

## Installation

```
python -m venv venv
pip install -r requirements
```

## Usage

Our algorithm supports 3 different structure of QBAF, baseline QBAF, QBAF with direct connections, and QBAF with 2 hidden layers. The structure of QBAFs are described in our paper. 

To run an algorithm

```
python grid_search_direct.py
```




