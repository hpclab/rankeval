#!/usr/bin/env bash

set -e -x

PACKAGE=rankeval
BASE_PYTHON=/opt/python/cp37-cp37m/bin/

${BASE_PYTHON}/pip install -U twine

cd /io/

# Compile wheels
for PYBIN in /opt/python/*/bin; do

    ${PYBIN}/pip install -U pip

    # install requirement sequentially
    cat build_tools/requirements.txt | xargs -L 1 ${PYBIN}/pip install

    ${PYBIN}/python setup.py clean
    ${PYBIN}/python setup.py sdist -d dist/
    ${PYBIN}/pip wheel ./ -w wheelhouse/
done

# Bundle external shared libraries into the wheels and fix naming
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w dist/
done

# Test
for PYBIN in /opt/python/*/bin; do

    ${PYBIN}/pip install --no-index -f dist/ ${PACKAGE}
    (cd $HOME; ${PYBIN}/nosetests ${PACKAGE})
done

#  Upload
for WHEEL in dist/${PACKAGE}*; do
    ${BASE_PYTHON}/twine upload \
        --skip-existing \
        -u "${PYPI_USER}" -p "${PYPI_PASS}" \
        "${WHEEL}"
done