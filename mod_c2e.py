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
