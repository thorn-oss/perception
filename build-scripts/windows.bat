call conda create -n py37 -y python=3.7
call conda create -n py38 -y python=3.8
call conda create -n py39 -y python=3.9
call conda create -n py310 -y python=3.10
call activate py37 && pip install numpy cython && python setup.py sdist
call activate py37 && pip install numpy cython && python setup.py bdist_wheel
call activate py38 && pip install numpy cython && python setup.py bdist_wheel
call activate py39 && pip install numpy cython && python setup.py bdist_wheel
call activate py310 && pip install numpy cython && python setup.py bdist_wheel
