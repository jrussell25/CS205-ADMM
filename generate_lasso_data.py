#!/usr/bin/python


import numpy as np
import sys
try: from mkl_random import randn
except ImportError:
    from numpy.random import randn


"""
Produce synthetic lasso data as in Boyd section 11.2.
Data matrix will be Nexamples x Nfeatures and the true
solution with have nnz non-zero elements (should be small
relative to Nfeatures).
"""
Nexamples, Nfeatures, nnz, nprocs = sys.argv[1:]

Nexamples = int(Nexamples)
Nfeatures = int(Nfeatures)
nnz = int(nnz)
nprocs = int(nprocs)

if Nexamples%nprocs != 0:
    Nexamples = Nexamples - Nexamples%nprocs
    print("Number of examples is not evenly divisible by number of processors")
    print(f"Using {Nexamples} examples instead")


M = int(Nexamples/nprocs)
xtrue = np.zeros(Nfeatures)
xtrue[np.random.choice(np.arange(Nfeatures),size=nnz,replace=False)] = np.random.randn(nnz)

np.savetxt('data/xtrue.csv', xtrue)


for i in range(nprocs):
    
    A = randn(M,Nfeatures)
    A /= np.linalg.norm(A,2,axis=1, keepdims=True)
    b = A.dot(xtrue) + 1e-3*randn(M)
    np.savetxt(f"data/b{i}.csv", b)
    np.savetxt(f"data/A{i}.csv",A, delimiter=',')

