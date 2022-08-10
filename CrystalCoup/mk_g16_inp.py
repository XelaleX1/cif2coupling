#!/usr/bin/env python

import os
import sys
import numpy as np
import argparse as arg
import MDAnalysis as mda
from jinja2 import Environment, FileSystemLoader

from CrystalCoup import templates as temps

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def mk_eet_inp(monomer1, monomer2, **kwargs):

    # Defaults
    basename = kwargs.pop("basename", "dimer")
    template = kwargs.pop("template", "dimer_EET.j2")
    comdir = kwargs.pop("outdir", "G16")

    # Load templates
    tempath = os.path.dirname(temps.__file__)
    file_loader = FileSystemLoader(tempath)
    env = Environment(loader=file_loader)
    template = env.get_template(template)

    # Fill in template
    dimtemp = template.render(
                        nproc=kwargs['NProc'],
                        mem=kwargs['Mem'],
                        funct=kwargs['DFT'],
                        basis=kwargs['Basis'],
                        disp=kwargs['Disp'],
                        dimername=basename,
                        mol1=monomer1,
                        mol2=monomer2,
                        render=False
                    )

    # Name and write
    dimfile = os.path.join(comdir, f'{basename}_EET.com')

    with open(dimfile, "w") as f:
        f.write(dimtemp)

    return dimfile


def mk_teet_inp(monomer1, monomer2, **kwargs):

    # Defaults
    basename = kwargs.pop("basename", "dimer")
    m1name = kwargs.pop("monomer1name", "monomer1")
    m2name = kwargs.pop("monomer2name", "monomer2")
    dimtemplate = kwargs.pop("template", "dimer_Tn.j2")
    montemplate = kwargs.pop("template", "monomer_Tn.j2")
    comdir = kwargs.pop("outdir", "G16")

    # Load templates
    tempath = os.path.dirname(temps.__file__)
    file_loader = FileSystemLoader(tempath)
    env = Environment(loader=file_loader)
    dimtemplate = env.get_template(dimtemplate)
    montemplate = env.get_template(montemplate)

    # Fill in templates
    dim1temp = dimtemplate.render(
                             nproc=kwargs['NProc'],
                             mem=kwargs['Mem'],
                             funct=kwargs['DFT'],
                             basis=kwargs['Basis'],
                             disp=kwargs['Disp'],
                             dimername=basename,
                             mol=np.r_[ monomer1, monomer2 ],
                             state=1,
                             render=False
                         )

    dim2temp = dimtemplate.render(
                             nproc=kwargs['NProc'],
                             mem=kwargs['Mem'],
                             funct=kwargs['DFT'],
                             basis=kwargs['Basis'],
                             disp=kwargs['Disp'],
                             dimername=basename,
                             mol=np.r_[ monomer1, monomer2 ],
                             state=2,
                             render=False
                         )

    m1temp = montemplate.render(
                            nproc=kwargs['NProc'],
                            mem=kwargs['Mem'],
                            funct=kwargs['DFT'],
                            basis=kwargs['Basis'],
                            disp=kwargs['Disp'],
                            monomername=m1name,
                            mol=monomer1,
                            state=1,
                            render=False
                        )

    m2temp = montemplate.render(
                            nproc=kwargs['NProc'],
                            mem=kwargs['Mem'],
                            funct=kwargs['DFT'],
                            basis=kwargs['Basis'],
                            disp=kwargs['Disp'],
                            monomername=m2name,
                            mol=monomer2,
                            state=1,
                            render=False
                        )

    # Name and write
    dim1file = os.path.join(comdir, f'{basename}_T1.com')
    dim2file = os.path.join(comdir, f'{basename}_T2.com')
    mon1file = os.path.join(comdir, f'{m1name}_T1.com')
    mon2file = os.path.join(comdir, f'{m2name}_T1.com')

    with open(dim1file, "w") as f:
        f.write(dim1temp)

    with open(dim2file, "w") as f:
        f.write(dim2temp)

    with open(mon1file, "w") as f:
        f.write(m1temp)

    with open(mon2file, "w") as f:
        f.write(m2temp)

    return dim1file, dim2file, mon1file, mon2file


def mk_ti_inp(monomer1, monomer2, **kwargs):

    # Defaults
    basename = kwargs.pop("basename", "dimer")
    m1name = kwargs.pop("monomer1name", "monomer1")
    m2name = kwargs.pop("monomer2name", "monomer2")
    dimtemplate = kwargs.pop("template", "dimer_TI.j2")
    montemplate = kwargs.pop("template", "monomer_TI.j2")
    comdir = kwargs.pop("outdir", "G16")

    # Load templates
    tempath = os.path.dirname(temps.__file__)
    file_loader = FileSystemLoader(tempath)
    env = Environment(loader=file_loader)
    dimtemplate = env.get_template(dimtemplate)
    montemplate = env.get_template(montemplate)

    # Fill in templates
    dimtemp = dimtemplate.render(
                            nproc=kwargs['NProc'],
                            mem=kwargs['Mem'],
                            funct=kwargs['DFT'],
                            basis=kwargs['Basis'],
                            disp=kwargs['Disp'],
                            dimername=basename,
                            mol=np.r_[ monomer1, monomer2 ],
                            render=False
                        )

    m1temp = montemplate.render(
                            nproc=kwargs['NProc'],
                            mem=kwargs['Mem'],
                            funct=kwargs['DFT'],
                            basis=kwargs['Basis'],
                            disp=kwargs['Disp'],
                            monomername=m1name,
                            mol=monomer1,
                            render=False
                        )

    m2temp = montemplate.render(
                            nproc=kwargs['NProc'],
                            mem=kwargs['Mem'],
                            funct=kwargs['DFT'],
                            basis=kwargs['Basis'],
                            disp=kwargs['Disp'],
                            monomername=m2name,
                            mol=monomer2,
                            render=False
                        )

    # Name and write
    dimfile = os.path.join(comdir, f'{basename}.com')
    mon1file = os.path.join(comdir, f'{m1name}.com')
    mon2file = os.path.join(comdir, f'{m2name}.com')

    with open(dimfile, "w") as f:
        f.write(dimtemp)

    with open(mon1file, "w") as f:
        f.write(m1temp)

    with open(mon2file, "w") as f:
        f.write(m2temp)

    return dimfile, mon1file, mon2file


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
                formatter_class=arg.ArgumentDefaultsHelpFormatter)

    #
    # Calculation Options
    #
    calc = parser.add_argument_group("Calculation Data")

    calc.add_argument('-n', '--nproc',
            type=int,
            default=8,
            dest='NProc',
            help='''Number of processors for G16 calculations.'''
        )

    calc.add_argument('-m', '--mem',
            type=int,
            default=24,
            dest='Mem',
            help='''Memory (in GB) for G16 calculations.'''
        )

    calc.add_argument('-f', '--funct',
            type=str,
            default='B3LYP',
            dest='DFT',
            help='''Density Functional for G16 calculations.'''
        )

    calc.add_argument('-b', '--basis',
            type=str,
            default='6-31G(d)',
            dest='Basis',
            help='''Basis set for G16 calculations.'''
        )

    calc.add_argument('-d', '--disp',
            type=str,
            default='GD3',
            dest='Disp',
            help='''Empirical Dispersion for G16 calculations.'''
        )

    calc.add_argument('-j', '--coup',
            type=str,
            default='TI',
            nargs='+',
            choices=[ 'TI', 'EET', 'TEET' ],
            dest='Coup',
            help='''Type of electronic coupling to compute.'''
        )

    #
    # Output Options
    #
    out = parser.add_argument_group("Output Options")

    out.add_argument('-o', '--out',
            default='G16',
            type=str,
            dest='Out',
            help='''Output folder.'''
        )

    args = parser.parse_args()
    Opts = vars(args)

    return Opts


def main():

    Opts = options()
    pwd = os.getcwd()
    comdir = os.path.join(pwd, Opts['Out'])
    try:
        os.makedirs(comdir)
    except:
        pass

    xyzs = [ i for i in os.listdir(pwd) if i.endswith(".xyz") ]
    for xyz in xyzs:
        fxyz = os.path.join(pwd, xyz)

        d = mda.Universe(fxyz)
        dim = np.c_[ d.atoms.types, d.atoms.positions ]
        n = int(dim.shape[0] / 2)
        m1 = dim[:n]
        m2 = dim[n:]

        # Excitonic coupling S-S
        if "EET" in Opts['Coup']:
            mk_eet_inp(m1, m2, **Opts)

        # Excitonic coupling T-T
        if "TEET" in Opts['Coup']:
            mk_teet_inp(m1, m2, **Opts)

        # Transfer Integrals
        if "TI" in Opts['Coup']:
            mk_ti_inp(m1, m2, **Opts)

    return


if __name__ == '__main__':
    main()
