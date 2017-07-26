This is the README.md of RankEval project.

copyright: HPCLab, ISTI-CNR, Pisa, Italy.

## Documentation

Api documentation is available here:
http://rankeval.isti.cnr.it/docs/

## Installation

The library works with OpenMP so you need a compiler supporting it. 
If your machine uses a default compiler different from gnu, change it 
appropriately before proceeding with the installation:

```
export CC=gcc-5
export CXX=g++-5
```

Below an example of installation.

```python setup.py install```

or

```pip install -e .```u

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


