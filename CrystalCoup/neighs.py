#!/usr/bin/env python

import os
import csv
import sys
import numpy as np
import pandas as pd
import networkx as nx
import argparse as arg
import MDAnalysis as mda
from MDAnalysis.lib import distances
from MDAnalysis.lib.util import unique_rows
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.analysis.distances import distance_array

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def map_unit_to_super(unitcell, supercell):
    '''
    '''

    u_in_s = {}
    for i, ifrag in enumerate(unitcell.atoms.fragments):
        for j, jfrag in enumerate(supercell.atoms.fragments):
            if np.allclose(ifrag.atoms.positions,
                           jfrag.atoms.positions):
                u_in_s[i] = j

    return u_in_s


def _find_contacts(fragments, cutoff=5.0):
    '''
    Raw version to return indices of touching fragments

    Parameters
    ----------
    fragments : list of AtomGroup
        molecules to consider
    cutoff : float
        threshold for touching or not

    Returns
    -------
    frag_idx : numpy array, shape (n, 2)
        indices of fragments that are touching, e.g. [[0, 1], [2, 3], ...]
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
    Calculate dimers to run

    Parameters
    ----------
    fragments : list of AtomGroups
        list of all fragments in system.  Must all be centered in box and
        unwrapped
    cutoff : float
        maximum distance allowed between fragments to be considered
        a dimer

    Returns
    -------
    dimers : dictionary
        mapping of {(x, y): (ag_x, ag_y)} for all dimer pairs
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
    check = np.allclose(d, d_nopbc, atol=tol)

    # if not, shift monomer2
    if not check:

        # compute the PBC shift vector
        shift = np.rint((com2 - com1) / box[:3])

        # apply the shift
        # newx = monomer2.positions - (np.rint(shift) * box[:3])
        newx = monomer2.positions - np.dot(shift, boxv)
        monomer2.positions = newx

    return monomer1, monomer2


def close_neighbours(dimers, cutoff=15.0):
    '''
    '''

    idxs = []
    for i, k in enumerate(list(dimers.keys())):

        # Get closest dimers according to Minimum Image Convention
        v = dimers[k]
        d = check_relative_position(v[0], v[1])

        # Distance in dimer 1
        com1 = d[0].center_of_mass()
        com2 = d[1].center_of_mass()
        r = np.linalg.norm(com1 - com2)

        if r <= cutoff:
            idxs.append(i)

    close_keys = [ list(dimers.keys())[i] for i in idxs ]
    close_dimers = { k: dimers[k] for k in close_keys }

    return close_dimers


def unique_neighbours(dimers, tol=1e-3):
    '''
    '''

    rdiff = np.zeros((len(list(dimers.keys())), len(list(dimers.keys()))))
    for i, k1 in enumerate(list(dimers.keys())):

        # Get closest dimers according to Minimum Image Convention
        v1 = dimers[k1]
        d1 = check_relative_position(v1[0], v1[1])

        # Distance in dimer 1
        com11 = d1[0].center_of_mass()
        com12 = d1[1].center_of_mass()
        r1 = np.linalg.norm(com11 - com12)

        for j, k2 in enumerate(list(dimers.keys())):#[i + 1:]):

            # Get closest dimers according to Minimum Image Convention
            v2 = dimers[k2]
            d2 = check_relative_position(v2[0], v2[1])

            # Distance in dimer 2
            com21 = d2[0].center_of_mass()
            com22 = d2[1].center_of_mass()
            r2 = np.linalg.norm(com21 - com22)

            # Compare distances
            rdiff[i,j] = np.abs(r1 - r2)

    # Select by non-equivalence
    G = nx.Graph()
    G.add_nodes_from(list(dimers.keys()))

    i, j = np.where(rdiff > tol)
    dim_i = [ list(dimers.keys())[x] for x in i ]
    dim_j = [ list(dimers.keys())[y] for y in j ]
    G.add_weighted_edges_from(zip(dim_i, dim_j, rdiff[i,j]), weight="r")
    subgraphs = [ G.subgraph(g) for g in nx.connected_components(G) ]
    ukeys = []
    for g in subgraphs:
        for edge in g.edges:
            ukeys.extend(edge)

    ukeys = sorted(list(set(ukeys)), key=lambda x: (x[0], x[1]))
    neighs = { k: dimers[k] for k in ukeys }

    return neighs


def miller_idxs(unitcell, monomer, box=None, tol=1e-3):
    '''
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
    for i, ifrag in enumerate(unitcell.atoms.fragments):
        if np.allclose(ifrag.atoms.positions,
                       newx):
            idx = i

    return idx, shift


def save_dimers(unitcell, supercell, dimers, outdir):

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
        idx1, shift1 = miller_idxs(unitcell, v[0])
        idx2, shift2 = miller_idxs(unitcell, v[1])
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
    u = mda.Universe(Opts['UnitFile'], guess_bonds=True)
    if Opts['UnitBox'] is not None:
        u.dimensions = Opts['UnitBox']
        u.atoms.dimensions = Opts['UnitBox']

    # Read Supercell
    s = mda.Universe(Opts['SuperFile'], guess_bonds=True)
    if Opts['SuperBox'] is not None:
        s.dimensions = np.r_[ Opts['UnitBox'][:3] * Opts['SuperBox'], Opts['UnitBox'][3: ] ]
        s.atoms.dimensions = np.r_[ Opts['UnitBox'][:3] * Opts['SuperBox'], Opts['UnitBox'][3: ] ]

    # Get map between mols in unit and super cells
    map_unit_super = map_unit_to_super(u, s)

    # Find all dimers in the supercell
    dimers = find_dimers(s.atoms.fragments, cutoff=Opts['Cutoff'])

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
    uneighs = unique_neighbours(close)

    # Save geometries
    coupdir, csvfile = save_dimers(u, s, uneighs, Opts['Out'])

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
            dest='SuperBox',
            help='''Miller indices size of the Super Cell.'''
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
