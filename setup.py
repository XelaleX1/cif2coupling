#!/usr/bin/env python

import setuptools
from numpy.distutils.core import Extension, setup

setup(
    name="CrystalCoup",
    version="1.0",
    author="Daniele Padula, Alessandro Landi",
    author_email="dpadula85@yahoo.it",
    description="A python package to compute electronic couplings in crystals",
    url="https://github.com/XelaleX1/cif2coupling",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPL License",
        "Operating System :: OS Independent",
    ],
    scripts=[
            'CrystalCoup/bin/process_crystal',
        ],
    entry_points={ 
        'console_scripts' : [
            'find_dimers=CrystalCoup.neighs:main',
            'cut_chains=CrystalCoup.cut_chains:main',
            'pack_crystal=CrystalCoup.pack_crystal:main',
            'make_coup_inp=CrystalCoup.mk_g16_inp:main',
        ]
    },
    zip_safe=False
)
