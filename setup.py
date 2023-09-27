#!/usr/bin/env python

from setuptools import setup

__author__ = 'Jared Markowitz, Ted Staley, Gino Perrotta'
__version__ = '0.1'

setup(
    name='mp2',
    version=__version__,
    description='Library for risk-sensitive, balanced multi-task, meta-learning',
    long_description=open('README.md').read(),
    author=__author__,
    author_email='jared.markowitz@jhuapl.edu',
    license='BSD',
    packages=['ntlb'],
    keywords='ntlb',
    classifiers=[],
    install_requires=['numpy', 'torch', 'torchvision', 'gym', 'scipy', 'tensorboard', 'mpi4py']
)
