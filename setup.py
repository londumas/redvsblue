#!/usr/bin/env python

import glob

from setuptools import setup

scripts = glob.glob('bin/*')

description = "Redshift fitter from PCA and redshift priors"

exec(open('py/redvsblue/_version.py').read())
version = __version__

with open('requirements.txt') as f:
    REQUIRED = f.read().splitlines()

setup(name="redvsblue",
    version=version,
    description=description,
    url="https://github.com/londumas/redvsblue",
    author="Helion du Mas des Bourboux",
    author_email="helion331990'at'gmail.com",
    packages=['redvsblue'],
    package_dir = {'': 'py'},
    package_data = {'redvsblue': ['etc/']},
    install_requires=REQUIRED,
    test_suite='redvsblue.test.test_cor',
    scripts = scripts
    )
