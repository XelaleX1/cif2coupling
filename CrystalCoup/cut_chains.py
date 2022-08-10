#!/usr/bin/env python

import os
import numpy as np
import MDAnalysis as mda


def remake_dist(x1, x2, d=1.09):
    '''
    Function to change coordinates x2 to be placed at distance d from x1.

    Parameters
    ----------
    x1: np.array.
        x1 coordinates.
    x2: np.array.
        x2 coordinates.
    d: float.
        Distance (in Angstrom) to place capping hydrogen atoms.


    Returns
    -------
    newx2: np.array.
        New x2 coordinates.
    '''

    bdvec = x2 - x1
    bdvec = bdvec / np.linalg.norm(bdvec)
    newx2 = x1 + bdvec * d

    return newx2


def cut_alkyl_chains(u, outtraj=None, alkyl=True, ether=False, d=1.09):
    '''
    Function to crop alkyl side chains in a universe, optionally writing
    a cropped trajectory.

    Parameters
    ----------
    u: object.
        MDAnalysis Universe to be cropped.
    outtraj: str.
        Name of the trajectory file to be optionally written. The file is
        written only if this argument is passed.
    alkyl: bool.
        Whether side chains to crop are purely alkylic.
    ether: bool.
        Whether side chains to crop have an oxygen atom connected to the
        sp2 core.
    d: float.
        Distance (in Angstrom) to place capping hydrogen atoms.


    Returns
    -------
    n: object.
        MDAnalysis Universe with cropped side chains.
    '''

    # Get bonds
    try:
        bds = u.bonds.to_indices()
    except:
        u.guess_bonds()
        bds = u.bonds.to_indices()

    # Get connectivity, use -1 as a placeholder for empty valence
    conn = np.ones((len(u.atoms), 4)) * -1
    for bond in bds:
        at1, at2 = bond
        for j in np.arange(conn[at1].shape[0]):
            if conn[at1,j] == -1:
                conn[at1,j] = at2
                break

        for j in np.arange(conn[at2].shape[0]):
            if conn[at2,j] == -1:
                conn[at2,j] = at1
                break

    # Get heavy atoms
    heavy = np.where(u.atoms.types != "H")[0]

    # Get sp3 atoms
    sat = np.where(np.all(conn > -1, axis=1))[0]

    # Get O atoms
    oxy = np.where(u.atoms.types == "O")[0]

    # Alkyls or ether chain
    if alkyl:
        allcheck = sat
    elif ether:
        allcheck = np.concatenate([sat, oxy])
    else:
        allcheck = sat

    # Check all sp3 atoms
    keep = []
    delete = []
    for satat in allcheck:

        # check connectivity
        iconn = conn[satat]

        # filter H out from connected
        iconnheavy = iconn[np.in1d(iconn, heavy)]

        # check if all connected heavy atoms are sp3
        check = np.in1d(iconnheavy, sat).all()

        check2 = True
        if ether:
            for ic in iconnheavy:
                xconn = conn[int(ic)] 
                xconnheavy = xconn[np.in1d(xconn, heavy)]
                check2 = np.in1d(xconnheavy, allcheck).all()
                if check2 == False:
                    break

        # if yes, delete the sp3 atom and its H
        if check and check2:
            delete.append(satat)
            delete.extend(iconn[~np.in1d(iconn, heavy)])
        # if not, keep the sp3 atom and its connected sp3 atom,
        # to be replaced with H
        else:
            keep.append(satat)
            keep.extend(iconn[np.in1d(iconn, allcheck)])

    # Convert to int arrays
    keep = np.asarray(keep).astype(int)
    delete = np.asarray(delete).astype(int)

    # Get non sp3 atoms
    unsat = ~np.all(conn > -1, axis=1)

    # Set which saturated atoms to keep or delete
    # Set keep later, so terminal sp3 C will be available for replacement
    unsat[delete] = False
    unsat[keep] = True

    # Get terminal sp3 to be replaced with H
    torepl = np.intersect1d(keep, delete)

    # Get atom names and coordinates
    ats = u.atoms.types
    coords = u.atoms.positions

    # Replace terminal sp3 C with H and select atoms to keep
    ats[torepl] = "H"
    toreplace = ' '.join(list(map(str, torepl)))
    u.select_atoms(f'index {toreplace}').names = "H"
    u.select_atoms(f'index {toreplace}').types = "H"
    ats = ats[unsat]
    tokeep = np.where(unsat)[0]
    tokeeps = ' '.join(list(map(str, tokeep)))

    if outtraj is not None:
        sel = u.select_atoms(f'index {tokeeps}')
        with mda.Writer(outtraj, len(tokeep)) as W:
            for ts in u.trajectory:

                # Shorten bonds with replaced atoms
                for at in torepl:

                    # Get indices and coordinates of atoms involved in the bond
                    iconn = conn[at]
                    cat = int(iconn[np.in1d(iconn, keep)][0])
                    cx = coords[cat]
                    hx = coords[at]

                    # Shorten bond to a typical C-H distance
                    coords[at] = remake_dist(cx, hx, d=d)

                u.atoms.positions = coords
                W.write(sel)

    else:
        # Shorten bonds with replaced atoms
        for at in torepl:

            # Get indices and coordinates of atoms involved in the bond
            iconn = conn[at]
            cat = int(iconn[np.in1d(iconn, keep)][0])
            cx = coords[cat]
            hx = coords[at]

            # Shorten bond to a typical C-H distance
            coords[at] = remake_dist(cx, hx, d=d)

        u.atoms.positions = coords

    sel = u.select_atoms(f'index {tokeeps}')
    n = mda.Merge(sel)

    return n


if __name__ == '__main__':

    gros = [ i for i in os.listdir(os.getcwd()) if i.endswith(".gro") ]
    for gro in gros:
        basename = '.'.join(gro.split(".")[:-1])
        trj = mda.Universe(gro, guess_bonds=True)
        u = cut_alkyl_chains(trj)
        u.atoms.write(f'{basename}.xyz')
