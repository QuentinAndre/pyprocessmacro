# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(

    name='PyProcessMacro',

    version="1.0.13",

    packages=find_packages(),

    author="Quentin André",

    author_email="quentin.andre@insead.edu",

    description="A Python library for moderation, mediation and conditional process analysis. Based on Andrew F. Hayes Process Macro.",

    long_description=open('README.md', encoding="utf-8").read(),

    long_description_content_type='text/markdown',

    install_requires=["numpy", "matplotlib", "pandas", "scipy", "seaborn"],

    keywords=['mediation-analysis', 'statistics', 'process', 'plotting', 'data-science', 'data-analysis',
              'data-visualization', 'regression-models'],

    url='https://github.com/QuentinAndre/pyprocessmacro/',

    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved",
        "Natural Language :: English",
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.14"
    ],

    license="MIT",
    python_requires='>=3.14'
)
