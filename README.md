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

Before compiling our code you need to tell cmake where the xtensor headers live. 
On the Harvard cluster, where user created conda environments live in 
`~/.conda`, one should run 

```cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/your_xtensor_env_name```.

Example for running the generate_lasso_data.py:

```python3 generate_lasso_data.py 160 500 30 1```


# Dirtributed ADMM

In order to discuss distributed implementation of ADMM, we start with introducing the preliminaries: *Dual Ascent, Dual Decomposition and Augmented Lagrangian*, as the fundations to ADMM.

## Dual Ascent

Consider the equality-constrained convex optimization problem 
<p align="center">
 <img src="figures/opt_problem.png" height="50">
</p>

The corresponding Lagrangian dual function is defined as 

<p align="center">
 <img src="figures/dual_func.png" height="35">
</p>

Assuming strong duality holds, the primal optimizer can be recovered by first finding the dual optimizer as follows

<p align="center">
 <img src="figures/primal_recovery.png" height="35">
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
 <img src="figures/augmented_opt_problem.png" height="50">
</p>

As the added penalty term vanishes for all the feasible primal variables, this does not change the result of the optimization.

The dual decomposition update rule with augmented penalty is 

<p align="center">
 <img src="figures/augmented_dual_update.png" height="50">
</p>

Note that the dual update step size is now the penalty parameter. The reason of making this choice is to make sure that each iteration the pair of the primal and dual variable is dual feasible. The primal minimization problem is changed to the Lagrangian with the penalty term. This is referred to as *the method of multiplers*. The method converges under far more general conditions than dual ascent.

Reproductivity parameter:
Cases were run on Harvard cluster of CentOS Linux release 7.6.1810 with x86_64 Intel(R) Xeon(R) Gold 6134 16 cores CPU @ 3.20GHz