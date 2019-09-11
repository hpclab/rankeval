#!/usr/bin/env bash

# This script is meant to be called by the "script" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project and by the setup.sh
# script which configure the environment.

set -x

cd $BUILD_DIRECTORY

if [[ $TASK == "sdist" ]]; then
    python setup.py sdist || exit -1
    pip install --user dist/rankeval-${RANKEVAL_VER}.tar.gz -v || exit -1
    python setup.py test || exit -1
    exit 0
elif [[ $TASK == "bdist" ]]; then
    python setup.py bdist_wheel || exit -1
    pip install --user $BUILD_DIRECTORY/dist/*.whl || exit -1
    python setup.py test || exit -1
    exit 0
fi