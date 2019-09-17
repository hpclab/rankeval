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
import shutil
import contextlib

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean as Clean
from distutils import log as _log
from distutils.dir_util import mkpath as _mkpath
import tempfile as _tempfile
from distutils.errors import CCompilerError as _CCompilerError
from distutils.version import LooseVersion
from distutils import log

min_cython_ver = '0.25.2'
try:
    import Cython
    ver = Cython.__version__
    _CYTHON_INSTALLED = ver >= LooseVersion(min_cython_ver)
except ImportError:
    _CYTHON_INSTALLED = False


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:

    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())
        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


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

    def _check_openmp_support(self):
        # Compile a test program to determine if compiler supports OpenMP.
        _mkpath(self.build_temp)
        with stdchannel_redirected(sys.stderr, os.devnull):
            with _tempfile.NamedTemporaryFile(mode='w',
                                              dir=self.build_temp,
                                              prefix='openmptest',
                                              suffix='.c') as srcfile:
                _log.info("checking if compiler supports OpenMP")
                srcfile.write("""
                #include <omp.h>
                int testfunc() {
                    int i;
                    #pragma omp parallel for
                    for (i = 0; i < 10; i ++)
                        ;
                    return omp_get_num_threads();
                }
                """)
                srcfile.flush()
                try:
                    objects = self.compiler.compile([srcfile.name],
                                                    extra_postargs=["-fopenmp"],
                                                    output_dir="/")
                except _CCompilerError:
                    _log.info("compiler does not support OpenMP")
                    use_openmp = False
                else:
                    _log.info("enabling OpenMP support")
                    use_openmp = True
                    for o in objects:
                        os.remove(o)
            return use_openmp

    def build_extensions(self):
        use_openmp = self._check_openmp_support()
        if use_openmp:
            for ext in self.extensions:
                if not ext.extra_compile_args:
                    ext.extra_compile_args = []
                ext.extra_compile_args.append('-fopenmp')
                if not ext.extra_link_args:
                    ext.extra_link_args = []
                ext.extra_link_args.append('-fopenmp')

        # Add language level directive (for the cython compiler)
        for ext in self.extensions:
            ext.compiler_directives = {'language_level' : sys.version_info[0]}

        # Chain to method in parent class.
        build_ext.build_extensions(self)
        if not _CYTHON_INSTALLED:
            log.info('No supported version of Cython installed. '
                     'Installing from compiled files.')
            for extension in self.extensions:
                sources = []
                for sfile in extension.sources:
                    path, ext = os.path.splitext(sfile)
                    if ext in ('.pyx', '.py'):
                        if extension.language == 'c++':
                            ext = '.cpp'
                        else:
                            ext = '.c'
                        sfile = path + ext
                    sources.append(sfile)
                extension.sources[:] = sources


# Custom clean command to remove build artifacts
class CleanCommand(Clean):
    description = "Remove build artifacts from the source tree"

    def run(self):
        Clean.run(self)
        # Remove c files if we are not within a sdist package
        cwd = os.path.abspath(os.path.dirname(__file__))
        remove_c_files = not os.path.exists(os.path.join(cwd, 'PKG-INFO'))
        if remove_c_files:
            print('Will remove generated .c files')
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('rankeval'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       (".so", ".pyd", ".dll", ".pyc")):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                extension = os.path.splitext(filename)[1]
                if remove_c_files and extension in ['.c', '.cpp']:
                    pyx_file = str.replace(filename, extension, '.pyx')
                    if os.path.exists(os.path.join(dirpath, pyx_file)):
                        os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


root_dir = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
# make it relative in order to avoid errors of absolute path by pypi
root_dir = os.path.relpath(root_dir)
rankeval_dir = os.path.join(root_dir, 'rankeval')
dataset_dir = os.path.join(rankeval_dir, 'dataset')
scoring_dir = os.path.join(rankeval_dir, 'scoring')
analysis_dir = os.path.join(rankeval_dir, 'analysis')

cmdclass = {
    'build_ext': custom_build_ext,
    'clean': CleanCommand
}

version = io.open(os.path.join(rankeval_dir, 'VERSION'),
                  encoding='utf-8').read().strip()
readme = io.open(os.path.join(root_dir, 'README.md'), encoding='utf-8').read()

setup(
    name='rankeval',
    version=version,
    description='Tool for the analysis and evaluation of Learning to Rank '
                'models based on ensembles of regression trees.',
    long_description=readme,
    long_description_content_type='text/markdown',

    ext_modules=[
        Extension('rankeval.dataset._svmlight_format',
                  sources=[dataset_dir + '/_svmlight_format.cpp'],
                  include_dirs=[dataset_dir],
                  language='c++',
                  extra_compile_args=['-O3', "-std=c++11"]),
        Extension('rankeval.scoring._efficient_scoring',
                  sources=[scoring_dir + '/_efficient_scoring.pyx'],
                  include_dirs=[scoring_dir],
                  extra_compile_args=['-O3'],
                  ),
        Extension('rankeval.analysis._efficient_topological',
                  sources=[analysis_dir + '/_efficient_topological.pyx'],
                  include_dirs=[analysis_dir],
                  extra_compile_args=['-O3'],
                  ),
        Extension('rankeval.analysis._efficient_feature',
                  sources=[analysis_dir + '/_efficient_feature.pyx',
                           analysis_dir + '/_efficient_feature_impl.cpp'],
                  include_dirs=[analysis_dir],
                  language='c++',
                  extra_compile_args=['-O3', "-w", "-std=c++11"],
                  )
    ],

    cmdclass=cmdclass,
    license='MPL 2.0',
    packages=find_packages(),

    author="HPC lab, ISTI-CNR",
    author_email='rankeval@isti.cnr.it',

    url='http://rankeval.isti.cnr.it',
    download_url='http://pypi.python.org/pypi/rankeval',

    keywords='Learning to Rank, Model Analysis, Model Evaluation, '
             'Ensemble of Regression Trees',

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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: System :: Benchmark'
    ],

    # py_modules=['svmlight_loader',],

    test_suite="rankeval.test",
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    setup_requires=[
        'Cython >= {}'.format(min_cython_ver),
        'setuptools >= 24.2.0',
        'numpy >= 1.13',
        'scipy >= 0.7.0'
    ],
    install_requires=[
        # Use 1.13: https://github.com/quantopian/zipline/issues/1808
        'numpy >= 1.13',
        'scipy >= 0.14.0',
        'six >= 1.9.0',
        'pandas >= 0.19.1',
        'xarray >= 0.10.9',
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
