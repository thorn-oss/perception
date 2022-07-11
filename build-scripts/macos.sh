#!/bin/bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
conda create -n py37 -y python=3.7
conda create -n py38 -y python=3.8
conda create -n py39 -y python=3.9
conda create -n py310 -y python=3.10
conda run -n py37 python setup.py sdist
for pyenv in py37 py38 py39 py310
do
	conda run -n $pyenv pip install numpy cython wheel
	conda run -n $pyenv python setup.py version
	conda run -n $pyenv python setup.py sdist
	conda run -n $pyenv pip wheel --no-deps --wheel-dir dist dist/*.tar.gz
done
