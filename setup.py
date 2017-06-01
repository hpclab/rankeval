#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 HPC lab, ISTI-CNR <salvatore.trani@isti.cnr.it>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Run with:

sudo python ./setup.py install
"""

import sys
import ez_setup
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if sys.version_info[:2] < (2, 7) or (sys.version_info[:1] == 3 and sys.version_info[:2] < (3, 5)):
    raise Exception('This version of gensim needs Python 2.7, 3.5 or later.')

ez_setup.use_setuptools()


class custom_build_ext(build_ext):
    # the following is needed to be able to add numpy's include dirs... without
    # importing numpy directly in this script, before it's actually installed!
    # http://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy as np
        self.include_dirs.append(np.get_include())


rankeval_dir = os.path.join(os.path.dirname(__file__), 'rankeval')
dataset_dir = os.path.join(rankeval_dir, 'core', 'dataset')
scoring_dir = os.path.join(rankeval_dir, 'core', 'scoring')

cmdclass = {'build_ext': custom_build_ext}

LONG_DESCRIPTION = u"""
==============================================
RankEval -- Analysis and evaluation of Learning to Rank models
==============================================
RankEval is a Python library for the #analysis# and #evaluation# of Learning to Rank models based on ensembles of regression trees.
Target audience is the *machine learning* (ML) and *information retrieval* (IR) communities.

Features
---------
* The tool is **memory-dependent** w.r.t. the corpus size (input dataset needs to fit in RAM),
* **Intuitive interfaces**
  * easy to load your own input corpus/model (supported the model format of the most popular learning tools such as QuickRank, RankLib, XGBoost, Scikit-Learn, etc.
  * easy to extend with other analysis and evaluations
* Efficient multicore implementations of the evaluation phase
* Extensive `documentation and Jupyter Notebook support for the presentation of the detailed analysis.

Installation
------------
This software depends on `NumPy and Scipy <http://www.scipy.org/Download>`_, two Python packages for scientific computing.
You must have them installed prior to installing `rankeval`.
The simple way to install `rankeval` is::
    pip install -U rankeval
Or, if you have instead downloaded and unzipped the `source tar.gz <http://pypi.python.org/pypi/rankeval>`_ package,
you'd run:
    python setup.py test
    python setup.py install
This version has been tested under Python 2.7 and 3.5.

How come RankEval is so fast and memory efficient? Isn't it pure Python, and isn't Python slow and greedy?
--------------------------------------------------------------------------------------------------------
Many scientific algorithms can be expressed in terms of large matrix operations. RankEval taps into these low-level BLAS libraries, by means of its dependency on NumPy. So while gensim-the-top-level-code is pure Python, it actually executes highly optimized Fortran/C under the hood, including multithreading (if your BLAS is so configured).
Memory-wise, gensim makes heavy use of Python's built-in generators and iterators for streamed data processing. Memory efficiency was one of gensim's `design goals <http://radimrehurek.com/gensim/about.html>`_, and is a central feature of gensim, rather than something bolted on as an afterthought.

Citing RankEval
-------------
When `citing RankEval in academic papers, please use this BibTeX entry::
  @inproceedings{
    TODO: ADD HERE!
  }
----------------
RankEval is open source software released under the `GNU LGPLv2.1 license <http://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html>`_.
Copyright (c) 20017-now HPC-ISTI CNR
"""

setup(
    name='rankeval',
    version='0.1',
    description='Tool for the analysis and evaluation of Learning to Rank '
                'models based on ensembles of regression trees.',
    long_description=LONG_DESCRIPTION,

    ext_modules=[
        Extension('rankeval.core.dataset._svmlight_format',
                  sources=[dataset_dir + '/_svmlight_format.cpp'],
                  include_dirs=[dataset_dir],
                  extra_compile_args=['-O3']),
        Extension('rankeval.core.scoring._efficient_scoring',
                  sources=[scoring_dir + '/_efficient_scoring.pyx'],
                  include_dirs=[scoring_dir],
                  extra_compile_args=['-fopenmp', '-O3'],
                  extra_link_args=['-fopenmp'],)
    ],

    cmdclass=cmdclass,
    packages=find_packages(),

    author="HPC lab, ISTI-CNR",
    author_email='rankeval@isti.cnr.it',

    url='http://rankeval.isti.cnr.it',
    download_url='http://pypi.python.org/pypi/rankeval',

    keywords='Learning to Rank, Model Analysis, Model Evaluation, Ensemble of Regression Trees',

    platforms='any',

    zip_safe=False,

    classifiers=[  # from http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Framework :: Jupyter',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Benchmark'
    ],

    # py_modules=['svmlight_loader',],

    test_suite="rankeval.test",
    setup_requires=[
        'numpy >= 1.3'
    ],
    install_requires=[
        'setuptools > 35',
        'setuptools_scm > 1.15',
        'six >= 1.9.0',
        'ez-setup >= 0.8',
        'numpy >= 1.12',
        'scipy >= 0.7.0',
        'sphinx >= 1.5.0',
        'sphinx_rtd_theme >= 0.2.0',
        'numpydoc > 0.5.0',
        'pandas > 0.19.1',
    ],
    # extras_require={
    #     'test': [
    #         'unittest2'
    #     ],
    # },

    include_package_data=True,
)
