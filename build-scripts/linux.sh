#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel python-devel

# This line is needed to allow Versioneer to get the version from Git.
git config --global --add safe.directory /io
git tag

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    # Block unsupported Python Versions. This currently blocks 2.7 and 3.11.
    if [[ ${PYBIN} != *$"cp27"* ]] && [[ ${PYBIN} != *$"cp311"* ]]; then
        echo ${PYBIN}
        "${PYBIN}/pip" install cython numpy
        "${PYBIN}/python" setup.py version
        "${PYBIN}/python" setup.py sdist
        "${PYBIN}/pip" wheel --no-deps --wheel-dir dist dist/*.tar.gz
    fi
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux2010_x86_64 -w /io/dist
done

rm dist/*linux_x86_64*.whl
