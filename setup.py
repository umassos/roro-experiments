# experiment implementations for online conversion with switching costs
# Cython build file
# August 2023

from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "functions.pyx", compiler_directives={"language_level": "3"}
    ),
    include_dirs=[numpy.get_include()]
)