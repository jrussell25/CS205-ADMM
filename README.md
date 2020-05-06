# CS205-ADMM
Group 17 final project for Harvard CS 205, Spring 2020.
 
### Required Software

This implementation requires on `xtensor` and the closely related library `xtensor-blas`. 
For optimal performance, one should also use `xsimd`.
The easier way to install these libraries is with Anaconda. The following command will
install these libraries and their dependencies (it is recommended to install these packages in
a fresh conda environment.

```conda install -c conda-forge xtensor xtensor-blas xsimd```

Before compiling our code you need to tell cmake where the xtensor headers live. 
On the Harvard cluster, where user created conda environments like in 
`~/.conda`, one should run 

```cmake -DCMAKE_INSTALL_PREFIX=~/.conda/envs/your_xtensor_env_name```.
