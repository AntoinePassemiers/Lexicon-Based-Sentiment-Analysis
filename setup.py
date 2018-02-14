# -*- coding: utf-8 -*-
# setup.py
# author : Antoine Passemiers

import os
from setuptools import setup, find_packages


datafiles = [(d, [os.path.join(d, f) for f in files]) 
    for d, folders, files in os.walk('data')]

setup(
    name='lbsa',
    version='1.0.0',
    author='Antoine Passemiers',
    description='Lexicon-based sentiment analysis',
    packages = find_packages('src'),
    include_package_data=True,
    package_dir={"": "src"},
    py_modules=["lbsa"],
    data_files = datafiles,
    url='https://github.com/AntoinePassemiers/Lexicon-Based-Sentiment-Analysis',
    install_requires=[
        'numpy >= 1.13.3',
        'matplotlib >= 2.0.2',
        'nltk >= 3.2.3'
    ],
)