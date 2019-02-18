from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='tennis-gp',
    version=getenv("VERSION", "LOCAL"),
    description='Predicts matches using a Gaussian Process',
    packages=find_packages()
)
