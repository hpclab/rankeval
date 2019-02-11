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
import os
import io

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

if (sys.version_info[:1] == 2 and sys.version_info[:2] < (2, 7)) or \
    (sys.version_info[:1] == 3 and sys.version_info[:2] < (3, 5)):
    raise Exception('This version of rankeval needs Python 2.7, 3.5 or later.')


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


root_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# make it relative in order to avoid errors of absolute path by pypi
root_dir = os.path.relpath(root_dir)
rankeval_dir = os.path.join(root_dir, 'rankeval')
dataset_dir = os.path.join(rankeval_dir, 'dataset')
scoring_dir = os.path.join(rankeval_dir, 'scoring')
analysis_dir = os.path.join(rankeval_dir, 'analysis')

cmdclass = {'build_ext': custom_build_ext}

version = io.open(os.path.join(rankeval_dir, 'VERSION'),
                  encoding='utf-8').read().strip(),
readme = io.open(os.path.join(root_dir, 'README.md'), encoding='utf-8').read()

setup(
    name='rankeval',
    version=open(os.path.join(rankeval_dir, 'VERSION')).read().strip(),
    description='Tool for the analysis and evaluation of Learning to Rank '
                'models based on ensembles of regression trees.',
    long_description=readme,
    long_description_content_type='text/markdown',

    ext_modules=[
        Extension('rankeval.dataset._svmlight_format',
                  sources=[dataset_dir + '/_svmlight_format.cpp'],
                  include_dirs=[dataset_dir],
                  language='c++',
                  extra_compile_args=['-O3']),
        Extension('rankeval.scoring._efficient_scoring',
                  sources=[scoring_dir + '/_efficient_scoring.pyx'],
                  include_dirs=[scoring_dir],
                  extra_compile_args=['-fopenmp', '-O3'],
                  extra_link_args=['-fopenmp'],),
        Extension('rankeval.analysis._efficient_topological',
                  sources=[analysis_dir + '/_efficient_topological.pyx'],
                  include_dirs=[analysis_dir],
                  extra_compile_args=['-fopenmp', '-O3'],
                  extra_link_args=['-fopenmp'], ),
        Extension('rankeval.analysis._efficient_feature',
                  sources=[analysis_dir + '/_efficient_feature.pyx',
                           analysis_dir + '/_efficient_feature_impl.cpp'],
                  include_dirs=[analysis_dir],
                  language='c++',
                  extra_compile_args=['-fopenmp', '-O3', "-w", "-std=c++11"],
                  extra_link_args=['-fopenmp'], )
    ],

    cmdclass=cmdclass,
    license='MPL 2.0',
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
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
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
        'setuptools >= 18.0',
        'numpy >= 1.13',
        'scipy >= 0.7.0',
        'cython >= 0.25.2'
    ],
    install_requires=[
        # Use 1.13: https://github.com/quantopian/zipline/issues/1808
        'numpy >= 1.13',
        'scipy >= 0.14.0',
        'six >= 1.9.0',
        'pandas >= 0.19.1',
        'xarray >= 0.9.5',
        'seaborn >= 0.8',
        'matplotlib >= 2.0.2',
    ],
    tests_require=[
        'nose >= 1.3.7',
    ],

    extras_require={
        'develop': [
            'sphinx >= 1.5.0',
            'sphinx_rtd_theme >= 0.2.0',
            'numpydoc > 0.5.0',
        ],
    },

    include_package_data=True,
)
