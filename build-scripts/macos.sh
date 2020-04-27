#!/bin/bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
conda create -n py36 -y python=3.6
conda create -n py37 -y python=3.7
conda create -n py38 -y python=3.8
conda run -n py36 python setup.py sdist
for pyenv in py36 py37 py38
do
	conda run -n $pyenv pip install numpy cython wheel
	conda run -n $pyenv python setup.py sdist
	conda run -n $pyenv pip wheel --no-deps --wheel-dir dist dist/*.tar.gz
done