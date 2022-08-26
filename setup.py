# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

# https://stackoverflow.com/questions/47905546/how-to-pass-python-package-to-spark-job-and-invoke-main-file-from-package-with-a
# https://stackoverflow.com/questions/36461054/i-cant-seem-to-get-py-files-on-spark-to-work
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='Group26_MovieLens',
    version='1.0.0',
    description='Package for first iteration of final project',
    long_description=readme,
    author='group26',
    author_email='ga947@nyu.edu',
    url='https://github.com/nyu-big-data/final-project-group-26',
    packages=find_packages(exclude=('tests', 'scratch', 'data', 'dist', 'Group26_MovieLens.egg-info', 'build'))
)

