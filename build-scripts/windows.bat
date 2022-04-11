call conda create -n py37 -y python=3.7
call conda create -n py38 -y python=3.8
call conda create -n py39 -y python=3.9
call activate py36 && pip install numpy cython && python setup.py sdist
call activate py36 && pip install numpy cython && python setup.py bdist_wheel
call activate py37 && pip install numpy cython && python setup.py bdist_wheel
call activate py38 && pip install numpy cython && python setup.py bdist_wheel
