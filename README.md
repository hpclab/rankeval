[![Build Status](https://img.shields.io/travis/com/hpclab/rankeval/master.svg?logo=travis)](https://travis-ci.com/hpclab/rankeval)
[![Python version](https://img.shields.io/pypi/pyversions/rankeval.svg)](https://badge.fury.io/py/rankeval)
[![PyPI version](https://img.shields.io/pypi/v/rankeval.svg)](https://badge.fury.io/py/rankeval)
[![Wheel](https://img.shields.io/badge/wheels-%E2%9C%93-4c1.svg?longCache=true&logo=python&logoColor=white)](https://badge.fury.io/py/rankeval)
[![CPython Implementation](https://img.shields.io/pypi/implementation/rankeval.svg)](https://badge.fury.io/py/rankeval)
[![License](https://img.shields.io/badge/license-MPL%202.0-blue.svg)](https://badge.fury.io/py/rankeval)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3077136.3084140-blue)](https://doi.org/10.1145/3077136.3084140)

<img src="https://github.com/hpclab/rankeval/blob/master/doc/banner.png?raw=true">

# RankEval: An Evaluation and Analysis Framework for Learning-to-Rank Solutions

RankEval is an open-source tool for the analysis and evaluation of
Learning-to-Rank models based on ensembles of regression trees. The
success of ensembles of regression trees fostered the development of
several open-source libraries targeting efficiency of the learning phase
and effectiveness of the resulting models. However, these libraries offer
only very limited help for the tuning and evaluation of the trained models.

RankEval aims at providing a common ground for several Learning to Rank 
libraries by providing useful and interoperable tools for a comprehensive
comparison and in-depth analysis of ranking models. Target audience is the 
*machine learning* (ML) and *information retrieval* (IR) communities.

RankEval is available under Mozilla Public License 2.0.

The official GitHub repository is: [here](https://github.com/hpclab/rankeval).

For questions/suggestions on how to improve RankEval, send us an email: 
rankeval@isti.cnr.it

## Features

Rankeval provides a common ground between several pre-existing tools and offers 
services which support the interpretation of differently generated models in a 
unified environment, allowing an easy, comprehensive comparison and in-depth 
analysis.

The main functionalities of RankEval can be summarized along five dimensions:
- effectiveness analysis
- feature analysis
- structural analysis
- topological analysis
- interoperability among GBRT libraries

Regarding the interoperability, Rankeval is able to read and process ranking ensembles learned with learning-to-rank 
libraries such as QuickRank, RankLib, XGBoost, LightGBM, Scikit-Learn, CatBoost, JForest. This advanced 
interoperability is implemented through proxy classes that make possible to interpret and understand the specific 
format used to represent the ranking ensemble without using the codebase of the learning-to-rank library. Thus RankEval 
does not have any dependency from the learning-to-rank library of choice of the user.

These functionalities can be applied to several models at the same time, so to 
have a direct comparison of the analysis performed. The tool has been written 
to ensure **flexibility**, **extensibility**, and **efficiency**. 

## Documentation

The official API documentation is available at: [here](http://rankeval.isti.cnr.it/docs/).
Soon on ReadTheDocs!

## Installation

The library works with OpenMP so you need a compiler supporting it. 
If your machine uses a default compiler different from GNU GCC, change it 
appropriately before proceeding with the installation:

```
export CC=gcc-5
export CXX=g++-5
```

Moreover, RankEval needs the following libraries to be installed before the 
installation process begin (used for compiling the low-level code by the installation process):
  - numpy >= 1.13
  - scipy >= 0.14
  - cython >= 0.25
  - matplotlib >= 2.0.2

Additional dependencies will be installed automatically by setuptools.
RankEval can be installed from the source by running:

```python setup.py install```

RankEval can also be easily installed from Python Package Index (PyPI). In this case, most probably you don't even need 
cython locally to compile low level code since the binaries should already been available for your platform.  
You may download and install it by running:

```pip install rankeval```

Alternatively, you can build the library from the latest commit on the master branch of the repository.
Below an example of installation.

```pip install git+https://github.com/hpclab/rankeval```

## Development

If you would like to install the library in development mode, i.e., you can edit the source code and see the changes 
directly without having to reinstall every time that you make a little change, than you have to run the following 
command which will install as well the libraries required for development (documentation generation and unittests):

```pip install -e .[develop]```

Local installation of compiled libraries: 

```python setup.py build_ext -i```

Execution of unit tests:

```python setup.py test```

or (if you have nose already installed):

```nosetests -v```

## Cite RankEval

If you use RankEval, please cite us!

```
@inproceedings{rankeval-sigir17,
  author = {Claudio Lucchese and Cristina Ioana Muntean and Franco Maria Nardini and
            Raffaele Perego and Salvatore Trani},
  title = {RankEval: An Evaluation and Analysis Framework for Learning-to-Rank Solutions},
  booktitle = {SIGIR 2017: Proceedings of the 40th International {ACM} {SIGIR}
               Conference on Research and Development in Information Retrieval},
  year = {2017},
  location = {Tokyo, Japan}
}
```

## Credits
    - Dataset loader: https://github.com/deronnek/svmlight-loader
    - Query id implementation: https://github.com/mblondel/svmlight-loader/pull/6