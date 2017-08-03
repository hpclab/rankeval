<img src=doc/banner.png>

# RankEval: An Evaluation and Analysis Framework for Learning-to-Rank Solutions

RankEval is an open-source tool for the analysis and evaluation of
Learning-to-Rank models based on ensembles of regression trees. The
success of ensembles of regression trees fostered the development of
several open-source libraries targeting efficiency of the learning phase
and effectiveness of the resulting models. However, these libraries offer
only very limited help for the tuning and evaluation of the trained models.

RankEval aims at providing a common ground for several Learning to Rank 
libraries by providing useful and interoperable tools for a comprehensive
comparison and in-depth analysis of ranking models.

RankEval is available under Mozilla Public License 2.0.

The official GitHub repository is: [here](https://github.com/hpclab/rankeval).

For questions/suggestions on how to improve RankEval, send us an email: [here](rankeval@isti.cnr.it).

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
installation process begin:
  - numpy >= 1.13
  - scipy >= 0.7
  - cython >= 0.25
  - matplotlib >= 2.0.2

Below an example of installation.

```python setup.py install```

or

```pip install -e .```

## Development

Installation of libraries required for development (documentation generation and unittests):

```pip install -e .[develop]```

Local installation of compiled libraries: 

```python setup.py build_ext -i```

Execution of unit tests:

```python setup.py test```

or (if you have nose already installed):

```nosetests -v```

## Credits:
    - Dataset loader: https://github.com/deronnek/svmlight-loader
    - Query id implementation: https://github.com/mblondel/svmlight-loader/pull/6