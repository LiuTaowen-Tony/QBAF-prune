# Iterative Pruning to learn QBAF structure

This is a repo for iterative pruning to learn QBAF structure


## Algorithm

```
procedureITERATIVEPRUNING(D,M,η,p,s)
  while Sparsity of M < s do
    Train M on D using learning rate η with l2 regularisation α
    Compute the absolute value of all weights |wi|
    Sort the weights by |wi |
    Set the lowest p percent of weights to zero
    Set the pruned weight to be not trainable
  end while
end procedure
```
