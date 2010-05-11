from setuptools import setup, Extension

setup(name='pyresample',
      version='0.5.2',
      description='Resampling of remote sensing data in Python',
      author='Esben S. Nielsen',
      author_email='esn@dmi.dk',
      package_dir = {'pyresample': 'pyresample'},
      packages = ['pyresample'],      
      install_requires=['configobj', 'multiprocessing']
      )



