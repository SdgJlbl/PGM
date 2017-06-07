#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

version = '0.0.1'

setup(
    name='pgm',
    version=version,
    description="Toying with Bayesian Networks",
    long_description="",
    author='OrangeBoreal',
    author_email='...',
    url='https://github.com/OrangeBoreal/PGM',
    packages=find_packages(exclude=["tests"]),
    include_package_data=True,
    license="MIT",
    entry_points={
        'console_scripts': ['pgm=pgm.main:main'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
    ],
)
