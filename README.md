# Iterative Pruning to learn QBAF structure

This is a repo for iterative pruning to learn QBAF structure

1: Input: Training dataset D, MLP model M, initial learning rate η, pruning per- centage p, desired sparsity s, l2 regularisation parameter α
2: procedureITERATIVEPRUNING(D,M,η,p,s)
3: while Sparsity of M < s do
4: Train M on D using learning rate η with l2 regularisation α
5: Compute the absolute value of all weights |wi|
6: Sort the weights by |wi |
7: Set the lowest p percent of weights to zero
8: Set the pruned weight to be not trainable
9: end while
10: end procedure
