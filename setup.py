# -*- coding: utf-8 -*-
import warnings
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import setup
from Cython.Build import cythonize
import numpy

import versioneer

class BuildFailure(Exception):
    pass

class CatchableBuildExt(build_ext):
    def run(self):
        try:
            build_ext.run(self)
        except DistutilsPlatformError:
            raise BuildFailure()

    def build_extension(self, ext):
        try:
            build_ext.build_extension(self, ext)
        except (CCompilerError, DistutilsExecError, DistutilsPlatformError):
            raise BuildFailure()

try:
    setup(version=versioneer.get_version(),
          cmdclass={**versioneer.get_cmdclass(), 'build_ext': CatchableBuildExt},
          ext_modules=cythonize(
              "perception/**/extensions.pyx",
          ), include_dirs=[numpy.get_include()])
except BuildFailure:
    warnings.warn('Failed to build Cython extensions. They will not be available at runtime.')
    setup(version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass())
