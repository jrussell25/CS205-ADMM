# Installation guide for cmake lapack and openblas

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
