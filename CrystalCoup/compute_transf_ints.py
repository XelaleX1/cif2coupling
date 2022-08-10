#!/usr/bin/env python

import os
import sys
import csv
import numpy as np
import pandas as pd
import multiprocessing as mp
from cclib.parser.fchkparser import FChk

au2eV = 27.21138505
eV2wn = 8065.544005


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
    accpt = "monomer_" + "_".join(data[2:]) + ".fchk"
    donor = donor.replace("dimer", "monomer")

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

    Js = np.array([ int(data[-1]), int(data[1]), int(data[2]), pht, pet, cr ])

    return Js


def parallel_fn(fn, iterable, nproc=None):
    '''
    Function to execute a generic function in parallel applying it to all
    the elements of an iterable. The function fn should contain appropriate
    error handling to avoid mp.Pool to hang up.

    Parameters
    ----------
    fn: object.
        Python standalone function.
    iterable: iterable.
        Collection of elements on which fn should be applied.
    nproc: integer (default: None).
        Number of processors for the parallelisation. If not specified all
        available processors will be used.

    Returns
    -------
    data: list.
        List of the returns of function fn for each element of iterable.
    '''

    if not nproc:
        nproc = os.cpu_count()

    pool = mp.Pool(nproc)
    data = pool.map(fn, iterable)                                                                                                                     
    pool.close()
    pool.join()

    return data


if __name__ == '__main__':


    fmats = [ i for i in os.listdir(os.getcwd()) if i.endswith(".fmat.dat") ]
    # data = parallel_fn(compute_TIs, fmats)

    data = []
    for i, fmat in enumerate(fmats):
        try:
            d = compute_TIs(fmat)
            data.append(d)
        except:
            pass
        print(i, fmat)

    data = np.array(data)

    df = pd.DataFrame({
        "Time / ns": data[:,0] / 1000.0,
        "DonorIdx" : data[:,1].astype(int),
        "AccptIdx" : data[:,2].astype(int),
        "PHT / eV" : data[:,3],
        "PET / eV" : data[:,4],
        "CR / eV" : data[:,5]
        })

    df = df.sort_values([ "Time / ns", "DonorIdx", "AccptIdx" ],
            ascending=[ True, False, False ])

    df.to_csv("transf_ints.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
