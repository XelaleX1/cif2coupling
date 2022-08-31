#!/usr/bin/env python

import os
import sys
import numpy as np
from ccdc import io
import argparse as arg
from openbabel import pybel

# Suppress Openbabel warnings
pybel.ob.obErrorLog.StopLogging()


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
                formatter_class=arg.ArgumentDefaultsHelpFormatter)

    #
    # Input Options
    #
    inp = parser.add_mutually_exclusive_group(required=True)

    inp.add_argument('-c', '--cif',
            type=str,
            default=None,
            dest='CIFFile',
            help='''Crystal structure file.'''
        )

    inp.add_argument('-id', '--identifier',
            type=str, 
            dest='CSD_ID',
            help='''Cambridge Structural Database identifier.'''
        )

    box = parser.add_argument_group("Cell Data")
    box.add_argument('-sb', '--superbox',
            type=int,
            nargs=3,
            default=[ 3, 3, 3 ],
            dest='SuperBox',
            help='''Miller indices size of the Super Cell.'''
        )

    #
    # Output Options
    #
    out = parser.add_argument_group("Output Options")

    out.add_argument('-p', '--pref',
            default=None,
            type=str,
            dest='Prefix',
            help='''Output file prefix.'''
        )

    args = parser.parse_args()
    Opts = vars(args)

    if Opts['SuperBox'] is not None:
        Opts['SuperBox'] = np.asarray(Opts['SuperBox'])

    return Opts


def pack_crystal(**Opts):

    if Opts['CIFFile'] is not None:
        x = io.CrystalReader(Opts['CIFFile'])[0]
    else:
        crystal_reader = io.CrystalReader('CSD')
        x = crystal_reader.crystal(Opts['CSD_ID'])

    if Opts['Prefix'] is None and Opts['CIFFile'] is not None:
        Opts['Prefix'] = ".".join(Opts['CIFFile'].split(".")[:-1])
    elif Opts['Prefix'] is None and Opts['CIFFile'] is None:
        Opts['Prefix'] = Opts['CSD_ID']

    fname = Opts['Prefix']

    box = list(x.cell_lengths) + list(x.cell_angles)
    box = np.asarray(box)
    superbox = box.copy()
    superbox[:3] *= Opts['SuperBox']

    high = (Opts['SuperBox'] // 2).astype(int)
    low = -high.astype(int)

    # Pack Unit Cell
    unitboxfile = f"{fname}_unit.dat"
    np.savetxt(unitboxfile, box.reshape(1,-1), fmt="%8.4f")

    unitciffile = f"{fname}_unit.cif"
    unitxyzfile = f"{fname}_unit.xyz"
    packed = x.packing()
    with io.CrystalWriter(unitciffile) as w:
        w.write(packed)
    
    mol = next(pybel.readfile("cif", unitciffile))
    mol.write("xyz", unitxyzfile, overwrite=True)
    
    # Pack Super Cell
    superboxfile = f"{fname}_super.dat"
    np.savetxt(superboxfile, superbox.reshape(1,-1), fmt="%8.4f")

    superciffile = f"{fname}_super.cif"
    superxyzfile = f"{fname}_super.xyz"
    packed = x.packing((tuple(low), tuple(high)))
    with io.CrystalWriter(superciffile) as w:
        w.write(packed)
    
    mol = next(pybel.readfile("cif", superciffile))
    mol.write("xyz", superxyzfile, overwrite=True)

    return unitxyzfile, box, superxyzfile


def main():

    # Get command line options
    Opts = options()
    pack_crystal(**Opts)

    return


if __name__ == '__main__':
    pass
