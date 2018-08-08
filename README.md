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
    - support the model format of the most popular learning tools such as 
    QuickRank, RankLib, XGBoost, LightGBM, Scikit-Learn, etc

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
installation process begin:
  - numpy >= 1.13
  - scipy >= 0.14
  - cython >= 0.25
  - matplotlib >= 2.0.2
  
RankEval can be easily installed from Python Package Index (PyPI). 
You may download and install it by running:

```pip install rankeval```

Alternatively, you can build the library from source.
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