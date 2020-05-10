# CS205-ADMM
Group 17 final project for Harvard CS 205, Spring 2020.
 
### Required Software

This implementation requires on `xtensor` and the closely related library `xtensor-blas`. 
For optimal performance, one should also use `xsimd`.
The easier way to install these libraries is with Anaconda. The following command will
install these libraries and their dependencies (it is recommended to install these packages in
a fresh conda environment).

```conda install -c conda-forge xtensor xtensor-blas xsimd```

In addition to these libraries, we use CMake to compile our code and Intel's C++ compiler, 
Math Kernel Library, and MPI Implementation. This software is all available on the Harvard Cluster.
The following command will load them all using the LMod package manager: `source gogo_modules.sh`.

If we are not running on Harvard cluster, we need to install lapack, Openblas and cmake 3 manually. we can install the newest from source(https://cmake.org/download/) by typing:

```wget https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2.tar.gz```

and unzip by typing:
```tar -zxvf cmake-3.17.2```

then enter the cmake folder and install by typing:
```./bootstrap```
```make```
```make install```

For the Lapack, we can download at http://www.netlib.org/lapack/#_lapack_version_3_9_0_2, enter the lapack folder, make a new folder called 'build' by:

```mkdir build/```

enter the build/ folder and type:
```cmake```
```sudo make install /usr/bin```

For OpenBlas, we can download by:
```wget https://codeload.github.com/xianyi/OpenBLAS/tar.gz/v0.3.9```
enter the unzipped folder and type 
```make```
and
```sudo make PREFIX=/usr/local install```

Before compiling our code you need to tell cmake where the xtensor headers live. 
On the Harvard cluster, where user created conda environments live in 
`~/.conda`, one should run 

```cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/your_xtensor_env_name```.

and then run
```make```

to get the executable. Then you can run:

```mpirun -np 1 xtensor_lasso```

to execute. You can change 1 to other numbers for MPI application.

Example for running the generate_lasso_data.py:

```python3 generate_lasso_data.py 160 500 30 1```

# Description of problem and the need for HPC
ADMM is the abbreviation of alternating direction method of multipliers. It comes from the classical convex optimization problem, where we want to minimize f(x) subject to an equality constraint:
```sh
Minimize f(x) subject to Ax=b
```
A typical application for ADMM is the compressed sensing, where we want to compress an image to about 10% of its original size, and be able to reconstruct the original image with the compressed one. This problem can be written in L1 problem, where we want to

```sh
Minimize L1 norm ||x|| such that Ax=b.
```
Where the number of rows in A represents the number of pixels in the compressed image. (reference at http://www.pyrunner.com/weblog/2016/05/26/compressed-sensing-python/)
The following chart shows the running time of L1 problem on the harvard cluster of CentOS 7 system with 48 core Intel(R) Xeon(R) Platinum 8268 CPU @2.90GHz with sequential ADMM method:
| size of matrix A | running time (s) |
| ------ | ------ |
| (2000 , 2500) | 1 |
| (4000 , 5000) | 5 |
| (8000 , 10000) | 44 |
| (16000 , 20000) | 247 |
| (32000 , 40000) | 1824 |

If we want to compress an image of 320000 pixels(figure size around 3MB) into 32000 pixels (size around 0.3MB), it would take 1824 seconds to finish. We need high performance computing and parallelization to optimize the running time. 

Reproductivity parameter:
Cases were run on Harvard cluster of CentOS Linux release 7.6.1810 with x86_64 Intel(R) Xeon(R) Gold 6134 16 cores CPU @ 3.20GHz

# Description of solution and comparison with existing work on the problem

Here we use diferent degree of parallelization to accelerate the existing ADMM method. We firstly implemented sequential ADMM in C++, added MPI function, then used xtensor for numpy style math in C++. Then we used blas (basic linear algebra subprograms) and lapack(linear algebra package) for low level of parallelization.
We developed our code to run on Havard research computing cluster with intel MPI implementation for distributed memory processing and MKL for multithreaded linear algebra needed for the local optimization steps on each MPI processor. 
Our result has running time of 40 seconds for matrix A of (640000, 8000) with 64 nodes x 16 cores, which is more than 100x faster than the sequential code.

# Dirtributed ADMM

In order to discuss distributed implementation of ADMM, we start with introducing the preliminaries: *Dual Ascent, Dual Decomposition and Augmented Lagrangian*, as the fundations to ADMM.

## Dual Ascent

Consider the equality-constrained convex optimization problem 
<p align="center">
 <img src="figures/opt_problem.png" height="35">
</p>

The corresponding Lagrangian dual function is defined as 

<p align="center">
 <img src="figures/dual_func.png" height="25">
</p>

Assuming strong duality holds, the primal optimizer can be recovered by first finding the dual optimizer as follows

<p align="center">
 <img src="figures/primal_recovery.png" height="30">
</p>

The *dual ascent method* is inspired by this idea, following the two steps: (1) Update the dual variable by ascending in the dual gradient direction, which equals to the residual of the equlity constraint, (2) Update the primal variable by minimizing the Lagrangian, fixing the dual variable:

<p align="center">
 <img src="figures/dual_ascent.png" height="50">
</p>

## Dual Decomposition

Consider special problems where the objective function is *separable* with respect to a partition of the primal variable (which is often the case in machine learning problems where the objective function is a summation of local cost functions w.r.t. data sample):

<p align="center">
 <img src="figures/separable.png" height="35">
</p>

In this case, the Lagrangian can also be decomposed

<p align="center">
 <img src="figures/decomp_largrangian.png" height="35">
</p>

This decomposition leads to a separation of the update rule, which is referred to as *dual decomposition*

<p align="center">
 <img src="figures/dual_decomposition.png" height="50">
</p>

This form is desirable for distributed optimization.

## Augmented Lagrangians and the Method of Multipliers

The problem of dual decomposition is that it requires strict convexity and finiteness of the objective function for convergence, which is a quite strong assumption in reality that can be difficult to satisfy. The augmented Lagrangian method was developed to make dual ascent method more robust to yield convergence without satisfying those strict requirements. This is done by adding an additional quadratic penalty term to the original objective function

<p align="center">
 <img src="figures/augmented_opt_problem.png" height="35">
</p>

As the added penalty term vanishes for all the feasible primal variables, this does not change the result of the optimization.

The dual decomposition update rule with augmented penalty is 

<p align="center">
 <img src="figures/augmented_dual_update.png" height="50">
</p>

Note that the dual update step size is now the penalty parameter. The reason of making this choice is to make sure that each iteration the pair of the primal and dual variable is dual feasible. The primal minimization problem is changed to the Lagrangian with the penalty term. This is referred to as *the method of multiplers*. The method converges under far more general conditions than dual ascent.

# Dirtributed Lasso

In this project, we consider L1 regularized linear regression, which is a standard machine learning problem:

<p align="center">
 <img src="figures/lasso.png" height="20">
</p>

The associated ADMM update is as follows:

<p align="center">
 <img src="figures/lasso_admm_update.png" height="50">
</p>

For efficiency consideration, the matrix inversion is calculated at the beginning, and cached for later multiplication.

The following diagrams illustrate how to distributedly implement the above equations by partitioning the data across several nodes.

<p align="center">
 <img src="figures/data_partition.png" height="100">
</p>

<p align="center">
 <img src="figures/computation_graph.png" height="300">
</p>

This distributed implementation can be summarized into 4 key steps:

- **Initilization**: Each node reads in the local matrix data into its local memory, and initlize local deicison variables *x* and *u*.

- **Local optimization**: Each node solves its local optimization problem (in Lasso, this local optimization is a ridge regression).

- **Global aggregation**: All the nodes communicate their local variables for averaging and broadcast the results back to all the nodes. We use MPI AllReduce to accomplish this aggregation.

- **Synchronization**: Synchronization between nodes must be enforced for the correctness of the implementation: All the local variables must be updated before global aggregation, and the local updates must all use the latest global variable.


# Our parallel implementation

We used a hybrid multi-node multi-core computing with lazy computing for this distributed ADMM application. 

### OpenMPI

We used [OpenMPI](https://www.open-mpi.org/faq/?category=mpi-apps) for c++ to launch parallel nodes, each of which corresponds to a piece of training data partition. We use 

### Xtensor

We used [xtensor](https://github.com/xtensor-stack/xtensor), an opensource c++ library for numerical analysis with multi-dimensional array expressions, with an extensible expression system enabling lazy broadcasting, and numpy-like syntax for array manipulation. With this lazy computing feature, both efficiency and memory usage are improved, allowing feasibility of larger scale applications.

### Xtensor-blas

[xtensor-blas](https://github.com/xtensor-stack/xtensor-blas) is a BLAS extension to the xtensor library, which enables the capability to use the BLAS and LAPACK libraries for optimized high performance computing with multi-core capability.


# Dataset

We generate synthetic dataset for both training and testing (see [generate_lasso_data.py](https://github.com/jrussell25/CS205-ADMM/blob/master/generate_lasso_data.py)). Both feature vector and the coefficient matrix are generated as Gaussian random variables. The right-hand-side vector b is computed as the matrix multiplication of the feature and coefficient matrix then corrupted by Gaussian noise. This allows us to easily compare our solution with the ground truth to verify the correctness of our implementation and to tune hyper-parameters for quicker convergence and termination criterion.

# Performance evaluation







# References:

1. [Boyd, Stephen, et al. "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends® in Machine learning 3.1 (2011): 1-122.](https://stanford.edu/~boyd/papers/admm_distr_stats.html)

2. [xtensor documentation](https://xtensor.readthedocs.io/en/latest/)

3. [xtensor-blas documentation](https://xtensor-blas.readthedocs.io/en/latest/)






