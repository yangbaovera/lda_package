

from setuptools import setup, find_packages

setup(
      name = "lda_package",
      version = "0.1",
      author='Yang Bao, Wenlin Wu',
      url='https://github.com/yangbaovera/lda_package',
      description='Implementation of Latent Dirichlet Allocation',
      license='MIT',
     # packages=find_packages(),
      py_modules = ['lda_package'],
      install_requires=[],
      zip_safe=False
      )
