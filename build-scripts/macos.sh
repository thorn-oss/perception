#!/bin/bash
HOMEBREW_NO_AUTO_UPDATE=1 brew install llvm libomp
export CC=/usr/local/opt/llvm/bin/clang
export CXX=/usr/local/opt/llvm/bin/clang++
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp"
conda create -n py39 -y python=3.9
conda create -n py310 -y python=3.10
for pyenv in py39 py310
do
	conda run -n $pyenv pip install poetry
    conda run -n $pyenv python -m poetry self add poetry-dynamic-versioning
	conda run -n $pyenv python -m poetry build -f wheel
done
