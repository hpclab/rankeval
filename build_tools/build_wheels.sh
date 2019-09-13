#!/usr/bin/env bash

set -e -x

PACKAGE=rankeval
BASE_PYTHON=/opt/python/cp37-cp37m/bin/

${BASE_PYTHON}/pip install -U twine

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install -U pip cython numpy scipy matplotlib
    "${PYBIN}/python" /io/setup.py sdist -d /io/wheelhouse/
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels and fix naming
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w /io/wheelhouse/
done

# Test
for PYBIN in /opt/python/*/bin/; do
    "${PYBIN}/pip" install --no-index -f /io/wheelhouse ${PACKAGE}
    (cd /io/; "${PYBIN}/nosetests" ${PACKAGE})
done

#  Upload
for WHEEL in /io/wheelhouse/${PACKAGE}*; do
    "${BASE_PYTHON}/twine" upload \
        --skip-existing \
        -u "${PYPI_USER}" -p "${PYPI_PASS}" \
        "${WHEEL}"
done