#!/usr/bin/env python

import os
import sys
import numpy as np
import MDAnalysis as mda


def centroid(coords, masses=None):
    '''
    Function to compute the centre (or the centre of mass) of a set of
    coordinates.

    Parameters
    ----------
    coord: np.array (N,3).
        coordinates.
    masses: np.array (N) (default: None).
        masses.

    Returns
    -------
    com: np.array (3).
        centre (or centre of mass) of the set of coordinates.
    '''

    com = np.average(coords, axis=0, weights=masses)

    return com


def rmse(x, y):
    '''
    Function to compute the Root Mean Square Error between two sets of points.

    Parameters
    ----------
    x: np.array (N,M).
        set of points.
    y: np.array (N,M).
        set of points.

    Returns
    -------
    err: float.
        Root Mean Squared Error.
    '''

    err = np.sqrt(np.mean( (x - y)**2))

    return err


# These come from https://github.com/charnley/rmsd
def kabsch(coords, ref):
    '''
    Function to transform a set of coordinates to match a reference.
    The transformation is done through the Kabsch algorithm. The order
    of points does not matter, as they get reordered to do calculations.

    Parameters
    ----------
    coords: np.array (N,D).
        set of points.
    ref: np.array (N,D).
        reference of points.

    Returns
    -------
    transf: np.array (N,D)
        set of points transformed to match the reference points.
    '''

    # Check for equal dimensions
    assert len(coords) == len(ref)

    P = coords.copy()
    Q = ref.copy()

    # # Reorder input coordinates according to their distance from their
    # # centroids
    # P, pidxs = _reorder_com(coords)
    # Q, qidxs = _reorder_com(ref)

    # Compute centroids
    com1 = centroid(P)
    com2 = centroid(Q)

    # Translate coordinates in the origin
    P -= com1
    Q -= com2

    # Get the optimal rotation matrix
    U = _kabsch(P, Q)

    # Rotate P unto Q
    P = np.dot(P, U)

    # Translate P unto ref
    P += com2

    # # Reverse P order to the originary one
    # transf = np.zeros_like(P)
    # transf[pidxs] = P
    transf = P

    return transf, U

def _kabsch(P, Q):
    '''
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : np.array
        (N,D) matrix, where N is points and D is dimension.
    Q : np.array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    U : np.array
        Rotation matrix (D,D)
    '''

    # Computation of the covariance matrix
    C = np.dot(P.T, Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


if __name__ == '__main__':

    # Reference
    u = mda.Universe("azulene-opt-004.Opt.xyz")
    ref = u.atoms.positions

    # List of dimers
    dims = [ i for i in os.listdir("..") if i.startswith("dimer") ]
    for dim in dims:

        newname = ".".join(dim.split(".")[:-1]) + "_prj.xyz"
        x = mda.Universe(f"../{dim}", guess_bonds=True)

        ats = u.atoms.types.copy()
        coords = []
        coms = []
        for frag in x.atoms.fragments:
            tgt = frag.atoms.positions
            transf, U = kabsch(ref, tgt)
            err = rmse(transf, tgt)
            print(err)
            coords.extend(transf)
            com = centroid(transf, masses=u.atoms.masses)
            coms.append(com)

        coms = np.array(coms)
        ats = np.repeat(ats.reshape(1,-1), len(x.atoms.fragments), axis=0)
        ats = ats.reshape(-1)
        coords = np.array(coords)
        struct = np.c_[ ats, coords ]

        with open(newname, "w") as f:
            f.write("%d\n\n" % struct.shape[0])
            np.savetxt(f, struct, fmt="%3s %12.6f %12.6f %12.6f")

        struct = np.c_[ 35 * np.ones(coms.shape[0]), coms ]
        with open("coms_" + newname, "w") as f:
            f.write("%d\n\n" % coms.shape[0])
            np.savetxt(f, struct, fmt="%3s %12.6f %12.6f %12.6f")
