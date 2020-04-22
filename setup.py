# -*- coding: utf-8 -*-

from setuptools import setup
from Cython.Build import cythonize
import numpy

import versioneer

setup(version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      ext_modules=cythonize(
          "perception/benchmarking/extensions.pyx",
      ), include_dirs=[numpy.get_include()])