#!/usr/bin/env python

import glob

from setuptools import setup

scripts = glob.glob('bin/*')

description = "Redshift fitter from PCA and redshift priors"
exec(open('py/redvsblue/_version.py').read())
version = __version__

setup(name="redvsblue",
    version=version,
    description=description,
    url="https://github.com/londumas/redvsblue",
    author="Helion du Mas des Bourboux",
    author_email="helion331990@gmail.com",
    packages=['redvsblue'],
    package_dir = {'': 'py'},
    package_data = {'redvsblue': ['etc/']},
    install_requires=['numpy','scipy','healpy','fitsio',
        'numba','future','setuptools'],
    test_suite='redvsblue.test.test_cor',
    scripts = scripts
    )
