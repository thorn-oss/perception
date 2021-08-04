# -*- coding: utf-8 -*-
import os
import warnings
from distutils.command.build_ext import build_ext
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError
from setuptools import setup
from Cython.Build import cythonize
import numpy

import versioneer

# Figuring out build issues.
# YMMV on some platforms
# https://stackoverflow.com/questions/60712479/cython-openmp-in-osx-no-build
# os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
# os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
# os.environ["LDFLAGS"] = (os.environ.get("LDFLAGS", "") +
# " -Wl,-rpath,/usr/local/opt/libomp/lib -L/usr/local/opt/libomp/lib -lomp")


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
    setup(
        version=versioneer.get_version(),
        cmdclass={**versioneer.get_cmdclass(), "build_ext": CatchableBuildExt},
        ext_modules=cythonize(
            "perception/**/extensions.pyx",
        ),
        include_dirs=[numpy.get_include()],
    )
except BuildFailure as err:
    warnings.warn(
        str(err)
        + "\nFailed to build Cython extensions. They will not be available at runtime."
    )
    setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
