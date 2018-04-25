
from setuptools import setup, find_packages

setup(
      name = "lda_package",
      version = "1.0",
      author='Yang Bao, Wenlin Wu',
      url='https://github.com/yangbaovera/lda_package',
      description='Implementation of Latent Dirichlet Allocation',
      packages=find_packages(),
      py_modules = ['lda_package'],
      install_requires=[]
      )
