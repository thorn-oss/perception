import numpy as np
from Cython.Build import cythonize
from setuptools import setup

compiler_directives = {"language_level": 3, "embedsignature": True}

setup(
    ext_modules=cythonize(
        "perception/**/extensions.pyx",
        compiler_directives=compiler_directives,
    ),
    include_dirs=[np.get_include()],
)
