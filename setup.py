from setuptools import Extension
from setuptools import setup

setup(
    name='zig_sum',
    version='1.0.1',
    python_requires='>=3.7.15',
    build_zig=True,
    ext_modules=[Extension('zig_sum', ['sum.zig'])],
    setup_requires=['setuptools-zig'],
)