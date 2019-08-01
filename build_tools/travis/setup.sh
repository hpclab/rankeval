#!/usr/bin/env bash

# This script is meant to be called by the "install" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -x

# install the right compiler
if [[ $OS_NAME == "macos" ]]; then
    OS_CONDA="MacOSX"

    # Conda export 10.6 osx version and this results in the usage
    # of libc++ in place of libstdc++, casuing the failure of the compilation
    # https://github.com/conda/conda-build/issues/1269
    export MACOSX_DEPLOYMENT_TARGET=10.9

    if [[ $COMPILER == "gcc" ]]; then
        brew update
        brew install gcc@8
        brew link --overwrite gcc
    fi
else    # Linux
    OS_CONDA="Linux"
    sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
    sudo apt-get update -q
    sudo apt-get install -y gcc-8 g++-8
fi

# export the compiler to use
if [[ $COMPILER == "gcc" ]]; then
    export CXX=g++-8
    export CC=gcc-8
elif [[ $COMPILER == "clang" ]]; then
    export CXX=clang++
    export CC=clang
fi

# Fix ccache with osx
if [[ $OS_NAME == "macos" ]]; then

    brew update
    brew install ccache
    export PATH="/usr/local/opt/ccache/libexec:$PATH"

    export CC=ccache $(which $CC)
    export CXX=ccache $(which $CXX)
fi

# debug: show stats of ccache
ccache -s

# install conda and setup test environment
wget -q -O miniconda.sh https://repo.continuum.io/miniconda/Miniconda${PYTHON_VERSION:0:1}-latest-$OS_CONDA-x86_64.sh

bash miniconda.sh -b -p $CONDA
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q -y conda

# Useful for debugging any issues with conda
conda info -a

conda create -q -y -n $CONDA_ENV python=$PYTHON_VERSION numpy scipy cython

pip install --user coremltools scikit-learn

source activate $CONDA_ENV