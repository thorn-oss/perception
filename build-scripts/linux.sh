#!/bin/bash
set -e -x

# Install a system package required by our library
yum install -y atlas-devel

# This line is needed to allow Versioneer to get the version from Git.
git config --global --add safe.directory /io
git tag

# Compile wheels
for PYBIN in /opt/python/cp39-cp39/bin /opt/python/cp310-cp310/bin; do
    echo ${PYBIN}
    "${PYBIN}/pip" install poetry
    "${PYBIN}/python" -m poetry self add "poetry-dynamic-versioning[plugin]"
    "${PYBIN}/python" -m poetry build -f wheel
done

# Bundle external shared libraries into the wheels
for whl in dist/*.whl; do
    auditwheel repair "$whl" --plat manylinux_2_28_x86_64 -w /io/dist
done
