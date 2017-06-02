This is the README.md of RankEval project.

copyright: HPCLab, ISTI-CNR, Pisa, Italy.


## Installation

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

or

```nosetests -v```


## Credits:
    - Dataset loader: https://github.com/deronnek/svmlight-loader
        - Query id implementation: https://github.com/mblondel/svmlight-loader/pull/6


