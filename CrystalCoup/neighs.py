#!/usr/bin/env python

import os
import csv
import sys
import numpy as np
import pandas as pd
import argparse as arg
import MDAnalysis as mda
from MDAnalysis.lib import distances
from MDAnalysis.lib.util import unique_rows
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.analysis.distances import distance_array

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def fragment(universe, nats):
    '''
    Function to define fragments independently of MDAnalysis

    Parameters
    ----------
    universe: object.
        MDAnalysis universe.
    nats : int.
        Number of atoms in each fragment.

    Returns
    -------
    fragments : tuple.
        Molecules to consider as list of MDAnalysis atom selections.
    '''

    tot = universe.atoms.n_atoms
    fragments = []
    for i, j in zip(np.arange(0, tot, nats), np.arange(0, tot, nats)[1:] - 1):
        ag = universe.select_atoms("index %d:%d" % (i, j))
        fragments.append(ag)

    return tuple(fragments)


def map_unit_to_super(ufrags, sfrags):
    '''
    '''

    u_in_s = {}
    for i, ifrag in enumerate(ufrags):
        for j, jfrag in enumerate(sfrags):
            if np.allclose(ifrag.atoms.positions,
                           jfrag.atoms.positions):
                u_in_s[i] = j

    return u_in_s


def _find_contacts(fragments, cutoff=5.0):
    '''
    Function to obtain indices of touching fragments.

    Parameters
    ----------
    fragments : list.
        Molecules to consider as list of MDAnalysis atom selections.
    cutoff : float (default: 5.0).
        Threshold for touching or not (in Angstroem).

    Returns
    -------
    frag_idx : np.array (N,2).
        Indices of fragments that are touching, e.g. [[0, 1], [2, 3], ...]
    '''

    # indices of atoms within cutoff of each other
    idx = distances.self_capped_distance(sum(fragments).positions,
                                         max_cutoff=cutoff,
                                         box=fragments[0].dimensions,
                                         return_distances=False)

    # TODO: Can optimise if all fragments are same size
    nfrags = len(fragments)
    fragsizes = [ len(f) for f in fragments ]
    # translation array from atom index to fragment index
    translation = np.repeat(np.arange(nfrags), fragsizes)
    # this array now holds pairs of fragment indices
    fragidx = translation[idx]
    # remove self contributions (i==j)
    fragidx = fragidx[fragidx[:,0] != fragidx[:,1]]
    fragidx = unique_rows(fragidx)

    # make first entry always less than second, e.g. (2, 1) -> (1, 2)
    flipped = fragidx[:,0] > fragidx[:,1]
    fragidx[flipped] = np.fliplr(fragidx[flipped])

    return unique_rows(fragidx)


def find_dimers(fragments, cutoff=5.0):
    '''
    Function to obtain dimers to calculate.

    Parameters
    ----------
    fragments : list.
        Molecules to consider as list of MDAnalysis atom selections.
    cutoff : float (default: 5.0).
        Threshold for touching or not (in Angstroem).

    Returns
    -------
    dimers : dict.
        Mapping of {(x, y): (ag_x, ag_y)} for all dimers.
    '''

    fragidx = _find_contacts(fragments, cutoff)

    dimers = {(i, j): (fragments[i], fragments[j])
              for i, j in fragidx}

    return dimers


def check_relative_position(monomer1, monomer2, box=None, tol=1e-3):
    '''
    Function to get monomers coordinates according to the minimum image
    convention.

    Parameters
    ----------
    monomer1: object.
        MDAnalysis atom selection.
    monomer2: object.
        MDAnalysis atom selection.
    box: list or np.array (6).
        Box describing PBCs as a list of a, b, c, alpha, beta, gamma
        (lenghts in Angstroem, angles in degrees).
    tol: float.
        Tolerance for distance comparison (in Angstroem).

    Returns
    -------
    monomer1: object.
        MDAnalysis atom selection.
    monomer2: object.
        Shifted monomer2 MDAnalysis atom selection.
    '''

    com1 = monomer1.center_of_mass()
    com2 = monomer2.center_of_mass()

    if box is None:
        box = monomer1.dimensions

    boxv = triclinic_vectors(box).T

    # Compute distance with and without PBC
    d = distance_array(com1, com2, box=box)
    d_nopbc = distance_array(com1, com2)

    # Check whether the two are not the same
    check = np.isclose(d, d_nopbc, atol=tol)

    # if not, shift monomer2
    if not check:

        # compute the PBC shift vector
        shift = np.rint((com2 - com1) / box[:3])

        # apply the shift
        # newx = monomer2.positions - (np.rint(shift) * box[:3])
        newx = monomer2.positions - np.dot(shift, boxv.T)
        monomer2.positions = newx

    return monomer1, monomer2


def close_neighbours(dimers, cutoff=15.0):
    '''
    Function to retain only dimers where monomers are within a threshold.

    Parameters
    ----------
    dimers : dict.
        Mapping of {(x, y): (ag_x, ag_y)} for all dimers.
    cutoff : float (default: 5.0).
        Threshold for touching or not (in Angstroem).

    Returns
    -------
    close_dimers : dict.
        Mapping of {(x, y): (ag_x, ag_y)} for all dimers.
    '''

    idxs = []
    for i, k in enumerate(list(dimers.keys())):

        # Get closest dimers according to Minimum Image Convention
        v = dimers[k]
        d = check_relative_position(v[0], v[1])

        # Distance between monomers COMs
        com1 = d[0].center_of_mass()
        com2 = d[1].center_of_mass()
        r = np.linalg.norm(com1 - com2)

        if r <= cutoff:
            idxs.append(i)

    close_keys = [ list(dimers.keys())[i] for i in idxs ]
    close_dimers = { k: dimers[k] for k in close_keys }

    return close_dimers


def unique_neighbours(dimers, tol=0.05):
    '''
    Function to discard symmetry equivalent dimers.

    Parameters
    ----------
    dimers : dict.
        Mapping of {(x, y): (ag_x, ag_y)} for all dimers.
    tol : float (default: 0.1).
        Threshold for equivalence comparison.

    Returns
    -------
    neighs : dict.
        Mapping of {(x, y): (ag_x, ag_y)} for all non-equivalent dimers.
    '''

    equiv = np.zeros((len(list(dimers.keys())), len(list(dimers.keys()))))
    for i, k1 in enumerate(list(dimers.keys())):

        # Get closest dimers according to Minimum Image Convention
        v1 = dimers[k1]
        d1 = check_relative_position(v1[0], v1[1])

        for j, k2 in enumerate(list(dimers.keys())):

            # Get closest dimers according to Minimum Image Convention
            v2 = dimers[k2]
            d2 = check_relative_position(v2[0], v2[1])

            # Compare dimers
            equiv[i,j] = is_equivalent(d1, d2, tol=tol)

    # Select by non-equivalence
    uidxs = []
    for r in equiv:
        idx = np.where(r == 1)[0][0]
        uidxs.append(idx)

    ukeys = [ list(dimers.keys())[i] for i in uidxs ]
    ukeys = sorted(list(set(ukeys)), key=lambda x: (x[0], x[1]))
    neighs = { k: dimers[k] for k in ukeys }

    return neighs


def is_equivalent(d1, d2, tol=0.05):
    '''
    Function evaluate equivalence between two dimers. The equivalence is
    evaluated through two contributions, Sd and SIn
    The first contribution evaluates the distance between centers of mass
    of the monomers within the two dimers.
    This contribution is given by the absolute dot product between the two
    distance vectors (maximum when aligned), weighted by a negative exponential
    of the absolute difference between the norms of such two vectors (maximum
    when they are the same).
    The second contribution is given by a similarity index between principal
    axes of inertia, and it is given by the average of the absolute dot product
    between the three principal axes.
    The two contributions are then averaged and, if close to 1 within the
    provided tolerance, the dimers are considered equivalent.

    Additional geometric criteria comparing principal axes of inertia of
    monomers within dimers can be easily added for a stricter evaluation
    of equivalence.

    Parameters
    ----------
    d1 : list.
        List of MDAnalysis atom selections describing a dimer.
    d2 : list.
        List of MDAnalysis atom selections describing a dimer.
    tol : float (default: 0.05).
        Threshold for equivalence comparison.

    Returns
    -------
    check : bool.
        Whether dimers d1 and d2 are equivalent or not, within tol.
    '''

    # 1 - monomer COMs distances criterion
    com11 = d1[0].center_of_mass()
    com12 = d1[1].center_of_mass()
    rij = com11 - com12
    rijnorm = np.linalg.norm(rij)
    rij /= rijnorm

    com21 = d2[0].center_of_mass()
    com22 = d2[1].center_of_mass()
    rkl = com21 - com22
    rklnorm = np.linalg.norm(rkl)
    rkl /= rklnorm

    # Check orientation of monomer distances
    # To do: we are not checking rijnorm and rklnorm
    d = np.abs(np.dot(rij, rkl))
    w = np.exp(-np.abs(rijnorm - rklnorm))
    Sd = w * d

    # 2 - Inertia axes criterion
    dim1 = sum(d1)
    Iij = dim1.principal_axes()[::-1].T
    if np.linalg.det(Iij) < 0.0:
        Iij[:,-1] = -Iij[:,-1]

    dim2 = sum(d2)
    Ikl = dim2.principal_axes()[::-1].T
    if np.linalg.det(Ikl) < 0.0:
        Ikl[:,-1] = -Ikl[:,-1]

    SIn = np.abs(np.diag(np.dot(Iij.T, Ikl))).mean()

    # Total similarity
    S = np.mean([ Sd, SIn ])
    check = np.isclose(S, 1.0, atol=tol)

    return check


def miller_idxs(unitcell, ufrags, monomer, box=None, tol=1e-3):
    '''
    Function get Miller indices describing the position of a monomer in a
    supercell with respect to the unit cell.

    Parameters
    ----------
    unitcell: object.
        MDAnalysis atom selection.
    ufrags : list.
        Molecules to consider as list of MDAnalysis atom selections.
    monomer: object.
        MDAnalysis atom selection.
    box: list or np.array (6).
        Box describing PBCs as a list of a, b, c, alpha, beta, gamma
        (lenghts in Angstroem, angles in degrees).
    tol: float.
        Tolerance for distance comparison (in Angstroem).

    Returns
    -------
    idx: int.
        Index of the molecule in the unit cell corresponding to monomer.
    shift: np.array (3).
        Miller indices describing the position of monomer in the supercell
        with respect to the unit cell.
    '''

    com1 = unitcell.atoms.center_of_mass()
    com2 = monomer.center_of_mass()

    if box is None:
        box = unitcell.dimensions

    boxv = triclinic_vectors(box).T

    # Compute distance with and without PBC
    d = distance_array(com1, com2, box=box)
    d_nopbc = distance_array(com1, com2)

    # Check whether the two are not the same
    check = np.allclose(d, d_nopbc, atol=tol)

    # if not, shift monomer
    if not check:

        # compute the PBC shift vector
        shift = np.rint((com2 - com1) / box[:3])

    else:
        shift = np.zeros(3)

    # apply the shift
    newx = monomer.positions - np.dot(shift, boxv)

    # figure out monomer correspondence in the unitcell
    idx = None
    for i, ifrag in enumerate(ufrags):
        if np.allclose(ifrag.atoms.positions,
                       newx):
            idx = i

    return idx, shift


def save_dimers(unitcell, supercell, ufrags, dimers, outdir):

    # Save geometries for coupling calculations and summary
    try:
        pwd = os.getcwd()
        coupdir = os.path.join(pwd, outdir)
        os.makedirs(coupdir)
    except:
        pass

    # Prepare dimers geometries
    data = []
    for k, v in dimers.items():

        # Compure Miller indices and shift geometries according to
        # Minimum Image Convention
        idx1, shift1 = miller_idxs(unitcell, ufrags, v[0])
        idx2, shift2 = miller_idxs(unitcell, ufrags, v[1])
        m1, m2 = check_relative_position(v[0], v[1], supercell.dimensions)
        d = m1 + m2
        data.append([ k[0], k[1], idx1, idx2 ] + shift1.tolist() + shift2.tolist() )

        # Write dimer geometries
        dgro = os.path.join(coupdir, "dimer_%03d_%03d.gro" % k)
        dxyz = os.path.join(coupdir, "dimer_%03d_%03d.xyz" % k)
        d.write(dgro)
        dimer = np.c_[ d.atoms.types, d.atoms.positions ]
        with open(dxyz, "w") as f:
            f.write("%d\n\n" % len(dimer))
            np.savetxt(f, dimer, fmt="%-3s %12.8f %12.8f %12.8f")

    # Write recap data
    data = np.asarray(data)
    df = pd.DataFrame({
            "Idx1 Supercell" : data[:,0],
            "Idx2 Supercell" : data[:,1],
            "Idx1 Unit cell" : data[:,2],
            "Idx2 Unit cell" : data[:,3],
            "Miller Idx1 a" : data[:,4],
            "Miller Idx1 b" : data[:,5],
            "Miller Idx1 c" : data[:,6],
            "Miller Idx2 a" : data[:,7],
            "Miller Idx2 b" : data[:,8],
            "Miller Idx2 c" : data[:,9]
        })

    csvfile = os.path.join(coupdir, "couplings_summary.csv")
    df.to_csv(csvfile, index=False, quoting=csv.QUOTE_NONNUMERIC)

    return coupdir, csvfile


def process_dimers(**Opts):

    # Read Unit Cell
    if Opts['FragAts'] is None:
        u = mda.Universe(Opts['UnitFile'], guess_bonds=True)
        ufrags = u.atoms.fragments
    else:
        u = mda.Universe(Opts['UnitFile'])
        ufrags = fragment(u, Opts['FragAts'])

    if Opts['UnitBox'] is not None:
        u.dimensions = Opts['UnitBox']
        u.atoms.dimensions = Opts['UnitBox']

    # Read Supercell
    if Opts['FragAts'] is None:
        s = mda.Universe(Opts['SuperFile'], guess_bonds=True)
        sfrags = s.atoms.fragments
    else:
        s = mda.Universe(Opts['SuperFile'])
        sfrags = fragment(s, Opts['FragAts'])

    if Opts['SuperBox'] is not None:
        s.dimensions = np.r_[ Opts['UnitBox'][:3] * Opts['SuperBox'], Opts['UnitBox'][3:] ]
        s.atoms.dimensions = np.r_[ Opts['UnitBox'][:3] * Opts['SuperBox'], Opts['UnitBox'][3:] ]

    # Get map between mols in unit and super cells
    map_unit_super = map_unit_to_super(ufrags, sfrags)

    # Find all dimers in the supercell
    dimers = find_dimers(sfrags, cutoff=Opts['Cutoff'])

    # Pick first molecule in the unit cell and get its index in the supercell
    refidx = 0
    refidx = map_unit_super[refidx]

    # Keep only dimers containing that mol
    neighs = { k: v for k, v in dimers.items() if refidx in k }

    # Rearrange them to always have the reference mol first
    reordered = {}
    for k, v in neighs.items():
        if k[0] == refidx:
            reordered[k] = v
        else:
            reordered[k[::-1]] = v[::-1]

    # Get rid of distant ones
    close = close_neighbours(reordered, cutoff=Opts['Threshold'])

    # Get rid of equivalent dimers
    uneighs = unique_neighbours(close, tol=Opts['EqTol'])

    # Save geometries
    coupdir, csvfile = save_dimers(u, s, ufrags, uneighs, Opts['Out'])

    return coupdir, csvfile


def options():
    '''Defines the options of the script.'''

    parser = arg.ArgumentParser(
                formatter_class=arg.ArgumentDefaultsHelpFormatter)

    #
    # Input Options
    #
    inp = parser.add_argument_group("Input Data")

    inp.add_argument('-u', '--unitcell',
            type=str,
            dest='UnitFile',
            required=True,
            help='''Unit Cell structure file.'''
        )

    inp.add_argument('-s', '--supercell',
            type=str, 
            dest='SuperFile',
            required=True,
            help='''Super Cell structure file.'''
        )

    inp.add_argument('-ub', '--unitbox',
            type=float,
            nargs=6,
            dest='UnitBox',
            help='''Unit Cell parameters.'''
        )

    inp.add_argument('-sb', '--superbox',
            type=int,
            nargs=3,
            default=[ 3, 3, 3 ],
            dest='SuperBox',
            help='''Miller indices size of the Super Cell.'''
        )

    inp.add_argument('-f', '--frag',
            default=None,
            type=int,
            dest='FragAts',
            help='''Number of atoms per fragment, if fragmenting outside of
            MDAnalysis.'''
        )

    inp.add_argument('-c', '--cutoff',
            type=float,
            default=5.0,
            dest='Cutoff',
            help='''Cutoff for proximity search of dimers.'''
        )

    inp.add_argument('-r', '--maxr',
            type=float,
            default=15.0,
            dest='Threshold',
            help='''Maximum distance between centres of mass for dimers.'''
        )

    inp.add_argument('-e', '--eq',
            type=float,
            default=0.10,
            dest='EqTol',
            help='''Tolerance for equivalence evaluation.'''
        )

    #
    # Output Options
    #
    out = parser.add_argument_group("Output Options")

    out.add_argument('-o', '--out',
            default='couplings',
            type=str,
            dest='Out',
            help='''Output folder.'''
        )

    args = parser.parse_args()
    Opts = vars(args)

    if Opts['UnitBox'] is not None:
        Opts['UnitBox'] = np.asarray(Opts['UnitBox'])
    if Opts['SuperBox'] is not None:
        Opts['SuperBox'] = np.asarray(Opts['SuperBox'])

    return Opts


def main():

    # Get command line options
    Opts = options()
    process_dimers(**Opts)

    return


if __name__ == '__main__':
    pass
