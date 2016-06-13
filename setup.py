from distutils.core import setup, Extension
import numpy

mod = Extension('WeatherGenerator',
    include_dirs = [numpy.get_include()],
    sources = ['WeatherGenerator.c'],
)

setup (name = 'WeatherGenerator',
    version = '0.3',
    description = 'Yet another stochastic precipitation generator.',
    author = 'Aljoscha Rheinwalt',
    author_email = 'aljoscha.rheinwalt@uni-potsdam.de',
    ext_modules = [mod]
)
