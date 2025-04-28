import re
import os
from setuptools import setup, find_packages


def get_version(package):
    """Return package version as listed in `__version__` in `init.py`."""
    with open(os.path.join(package, '__init__.py')) as f:
        init_py = f.read()
    return re.search("__version__ = [\'\"]([^\'\"]+)[\'\"]", init_py).group(1)


version = get_version('torchhook')  # Get version from torchhook/__init__.py

setup(
    name='torchhook',
    version=version,  # Use the extracted version here
    author='Zaiyan Zhang',
    author_email='1@zzaiyan.com',
    description='TouchHook: A PyTorch hook management library',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/zzaiyan/TorchHook',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.0.0'
    ],
)
