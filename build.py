from Cython.Build import cythonize
import numpy as np


compiler_directives = {"language_level": 3, "embedsignature": True}


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                "perception/**/extensions.pyx", compiler_directives=compiler_directives
            ),
            "include_dirs": [np.get_include()],
        }
    )
