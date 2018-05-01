#!/usr/bin/env python
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
	Extension('utils.fast_cts', sources=['utils/fast_cts.pyx']),
	Extension('utils.hogupdatemv', sources=['utils/hogupdatemv.pyx']),
]

setup(
	name='async-rl-extensions',
	include_dirs=[np.get_include()],   
	ext_modules=cythonize(extensions)
)
