# Module 

import os
import sys
import csv
import numpy as np
import pandas as pd
import multiprocessing as mp
from cclib.parser.fchkparser import FChk

#####
# CONSTANTS
au2eV = 27.21138505
eV2wn = 8065.544005



###############################
# For coupling
def symm_mat(M):
    '''
    Function to symmetrize an upper- or lower diagonal matrix.

    Parameters
    ----------
    M: np.array (N,N).
        Matrix to be symmetrised.

    Returns
    -------
    M: np.array (N,N).
        Symmetrised matrix.
    '''

    M = M + M.T - np.diag(M.diagonal())

    return M


def read_fock_matrix(filename, thresh=1e-6):

    fmat = []
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                els = list(map(float, line.replace("D", "E").split()))
                fmat.extend(els)

    fmat = np.array(fmat)
    fmat[np.abs(fmat) < thresh] = 0.0
    sqshape = int(0.5 * (np.sqrt(1 + 8 * fmat.shape[0]) - 1))
    F = np.zeros((sqshape, sqshape))
    idxs = np.tril_indices(sqshape)
    F[idxs] = fmat

    return symm_mat(F)


def compute_TIs(dimer):

    data = dimer.split(".")[0].split("_")
    donor = "_".join(data[:2] + [ data[-1] ]) + ".fchk"
    accpt = "_".join(data[2:]) + ".fchk"

    ddata = FChk(donor).parse()
    dhomoidx = ddata.homos[0]
    dlumoidx = dhomoidx + 1
    dhomo = ddata.mocoeffs[0][dhomoidx]
    dlumo = ddata.mocoeffs[0][dlumoidx]

    adata = FChk(accpt).parse()
    ahomoidx = adata.homos[0]
    alumoidx = ahomoidx + 1
    ahomo = adata.mocoeffs[0][ahomoidx]
    alumo = adata.mocoeffs[0][alumoidx]

    F = read_fock_matrix(dimer)[:dhomo.shape[0],dhomo.shape[0]:]

    pht = np.dot(np.dot(dhomo, F), ahomo) * au2eV
    pet = np.dot(np.dot(dlumo, F), alumo) * au2eV
    cr = np.dot(np.dot(dhomo, F), alumo) * au2eV

    Js = np.array([ int(data[-1]), int(data[1]), int(data[3]), pht, pet, cr ])

    return Js


#########
# For TEET
def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(description='''Calculates Electronic Couplings
                            from a diabatisation procedure.''',
                            formatter_class=arg.ArgumentDefaultsHelpFormatter)

    #
    # Input files
    #
    inp = parser.add_argument_group("Input Data")

    # Monomer 1
    inp.add_argument('--mon1', default="monomer1.log", type=str, dest='Mon1File',
                     help='''Data for monomer 1.''')

    inp.add_argument('--selmon1', default=None, nargs='+', type=str, dest='SelMon1',
                     help='''Atom Selection for monomer 1. This can either be a list
                     or a file.''')

    # Monomer 2
    inp.add_argument('--mon2', default="monomer2.log", type=str, dest='Mon2File',
                     help='''Data for monomer 2.''')

    inp.add_argument('--selmon2', default=None, nargs='+', type=str, dest='SelMon2',
                     help='''Atom Selection for monomer 2. This can either be a list
                     or a file.''')

    # Dimer 1
    inp.add_argument('--dim1', default="dimer_s1.log", type=str, dest='Dim1File',
                     help='''Data for dimer state 1.''')

    inp.add_argument('--dim2', default="dimer_s2.log", type=str, dest='Dim2File',
                     help='''Data for dimer state 2.''')

    inp.add_argument('--seldim', default=None, nargs='+', type=str, dest='SelDim',
                     help='''Atom Selection for dimer. This can either be a list
                     or a file.''')

    inp.add_argument('--states', default=None, nargs='+', type=int, dest='States',
                     help='''States Selection for dimer. This can either be a list
                     or a file.''')

    #
    # Calculations Options
    #
    calc = parser.add_argument_group("Calculation Options")

    calc.add_argument('--coup', default=None, type=str, choices=['fcd', 'fsd'],
                      help='''Method of Calculation of the Electronic Coupling.
                      The choice is exclusive.''', dest='Coup', required=True)

    #
    # Output Options
    #
    out = parser.add_argument_group("Output Options")

    out.add_argument('-o', '--output', default=None, type=str, dest='OutFile',
                     help='''Output File.''')

    out.add_argument('-v', '--verbosity',
                     default=0, action="count", dest="Verb",
                     help='''Verbosity level''')

    args = parser.parse_args()
    Opts = vars(args)

    return Opts


def get_data(logfile, *args, prop='atomcharges'):
    '''
    Function to extract data from logfile.

    Parameters
    ----------
    logfile: str.
        Name of the QM calculation output file.
    args: tuple of ints.
        Indexes of states.
    prop: str.
        Name of the property to extract (cclib attribute name).

    Returns
    -------
    enes: np.array.
        Energies of states.
    prop: np.array.
        property extracted.
    '''

    data = ccopen(logfile).parse()
    enes = data.etenergies[list(args)]
    prop = getattr(data, prop)['mulliken']

    return enes, prop


def compute_coupling(*args):
    '''
    Function to compute electronic couplings according to the diabatisation
    scheme described in J. Chem. Phys. 2015, 142, 164107.

    Parameters
    ----------
    args: tuple.
        should contain the following data structure
    enes: np.array.
        Energies of adiabatic states.
    m: np.array.
        Property of nth diabatic state for mth monomer. Order by monomers,
        then by state.
    d: np.array.
        Property of nth adiabatic state.

    Returns
    -------
    J: np.array (N * (N - 1) / 2).
        Off diagonal elements of the Diabatic Hamiltonian (eV).
    '''

    enes = args[0]
    nstates = len(enes)
    mdata = args[1:1 + nstates]
    ddata = args[1 + nstates:]

    # Make shapes of monomers consistent with dimers
    mdata = list(mdata)
    for i, m in enumerate(mdata):
        padl = len(ddata[0]) - len(m)
        if i % 2 == 0:
            mdata[i] = np.row_stack((m, np.zeros(padl))).reshape(-1)
        else:
            mdata[i] = np.row_stack((np.zeros(padl), m)).reshape(-1)

    # Create rectangular matrices describing properties in the Adiabatic and
    # Diabatic bases - rows are atoms, columns are states
    qA = np.column_stack(ddata)
    qD = np.column_stack(mdata)

    # Compute the rotation from Adiabatic to Diabatic basis
    M = np.dot(qA.T, qD)
    U, S, Vt = np.linalg.svd(M)
    C = np.dot(U, Vt).T

    # Build Adiabatic Hamiltonian and apply the same transformation computed
    # above. Data from cclib are in wavenumbers
    HA = np.diag(enes)
    HD = C.dot(HA).dot(C.T)

    # Extract off diagonal elements and convert to eV
    idxs = np.triu_indices(HD.shape[0], k=1)
    J = HD[idxs] / eV2wn

    return J
