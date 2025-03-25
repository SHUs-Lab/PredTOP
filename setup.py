from setuptools.discovery import PackageFinder
from distutils.core import setup


find_packages = PackageFinder.find

setup(
  name='predtop',
  version='1.0.0',
  packages=find_packages(where='src'),
  package_dir={'': 'src'},
  python_requires='==3.9.20',
  install_require=[
    'torch==1.13.0+cu117',
    'torch_geometric==2.5.3',
    'numpy==1.24.4',
    'networkx==3.2.1',
  ]
)
