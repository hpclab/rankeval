#!/usr/bin/env bash

# This script is meant to be called by the "script" step defined in
# .travis.yml. See https://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project and by the setup.sh
# script which configure the environment.

set -x -e

PACKAGE=rankeval

cd $BUILD_DIRECTORY

pip install --user -U twine

if [[ $TASK == "sdist" ]]; then
    python setup.py sdist
    pip install --user dist/${PACKAGE}-${RANKEVAL_VER}.tar.gz -v
elif [[ $TASK == "bdist" ]]; then
    python setup.py bdist_wheel
    pip install --user $BUILD_DIRECTORY/dist/*.whl
fi

# Test
python setup.py test

#  Upload
for WHEEL in dist/${PACKAGE}*; do
    "twine" upload \
        --skip-existing \
        -u "${PYPI_USER}" -p "${PYPI_PASS}" \
        "${WHEEL}"
done