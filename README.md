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

In order to discuss distributed implementation of ADMM, we start with introducing two optimization schemes, known as dual Ascent and augmented Lagrangian, as the fundations to ADMM.

## Dual Ascent

Consider the equality-constrained convex optimization problem 
<p align="center">
 <img src="figures/opt_problem.png" height="50">
</p>

The corresponding Lagrangian dual function is defined as 

<p align="center">
 <img src="figures/dual_func.png" height="30">
</p>

Assuming strong duality holds, the primal optimizer can be recovered by first finding the dual optimizer as follows

<p align="center">
 <img src="figures/primal_recovery.png" height="30">
</p>

The *dual ascent method* is inspired by this idea, following the two steps: (1) Update the dual variable by ascending in the dual gradient direction, which equals to the residual of the equlity constraint, (2) Update the primal variable by minimizing the Lagrangian, fixing the dual variable:

<p align="center">
 <img src="figures/dual_ascent.png" height="30">
</p>



