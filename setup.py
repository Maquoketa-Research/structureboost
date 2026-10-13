"""Setup for the structureboost package.

Builds the extension modules from the .pyx sources (the shipped .c files are regenerated, not compiled as-is), so
source changes take effect. Environment switches:
  STB_PROFILE=1   compile with Cython profiling hooks (function-level times in cProfile); slower
  STB_OPENMP=0    build without OpenMP (parallel loops then run single-threaded)
"""
import os

import numpy
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

MODULES = ["graphs", "structure_dt", "structure_gb", "structure_gb_multi", "pdf_discrete", "pdf_group", "pdf_set",
           "prob_regr_unit", "prob_regressor", "coarsage", "pdf_conformal",
           "structure_dt_multi"]
PROFILE = os.environ.get("STB_PROFILE") == "1"
OPENMP = os.environ.get("STB_OPENMP", "1") == "1"
cflags = ["-O3"] + (["-fopenmp"] if OPENMP else [])
lflags = ["-fopenmp"] if OPENMP else []

extensions = [Extension(m, [f"structureboost/{m}.pyx"], include_dirs=[numpy.get_include()],
                        extra_compile_args=cflags, extra_link_args=lflags) for m in MODULES]

setup(packages=find_packages(include=["structureboost", "structureboost.*"]), ext_modules=cythonize(extensions, compiler_directives={"language_level": 3, "profile": PROFILE}, annotate=False))
