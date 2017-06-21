This is the README.md of RankEval project.

copyright: HPCLab, ISTI-CNR, Pisa, Italy.


## Installation

The library works with OpenMP so you need a compiler supporting it. Below an example of installation with
GCC and G++ (required a gnu compiler).

```CC=gcc-6 CXX=g++-6 python setup.py install```

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


