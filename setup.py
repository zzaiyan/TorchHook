from setuptools import setup, find_packages

setup(
    name='torchhook',
    version='0.1.7',
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