import numpy as np
from Cython.Build import cythonize
from setuptools import setup

compiler_directives = {"language_level": 3, "embedsignature": True}

ext_modules = cythonize(
    "perception/**/extensions.pyx",
    compiler_directives=compiler_directives,
)

for extension in ext_modules:
    extension.include_dirs = list(getattr(extension, "include_dirs", [])) + [np.get_include()]

setup(
    ext_modules=ext_modules,
)
