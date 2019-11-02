from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
  name = 'Urns model evolution.',
  ext_modules = cythonize("poliakov.pyx", annotate=True),
  include_dirs=[numpy.get_include(), '.', '../']
)
