call conda create -n py39 -y python=3.9
call conda create -n py310 -y python=3.10
call activate py39
call pip install poetry
call poetry self add "poetry-dynamic-versioning[plugin]"
call poetry build -f wheel
call activate py310
call pip install poetry
call poetry self add "poetry-dynamic-versioning[plugin]"
call poetry build -f wheel
