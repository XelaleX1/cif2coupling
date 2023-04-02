#!/usr/bin/env python

import numpy as np
import MDAnalysis as mda
from MDAnalysis.lib.mdamath import triclinic_vectors


if __name__ == '__main__':
    u = mda.Universe("azlene_unit.xyz")
    box = np.loadtxt("azlene_unit.dat")
    a, b, c = triclinic_vectors(box)
    ats = u.atoms.names
    xyz = u.atoms.positions

    final = []
    i = np.arange(-2, 3, 1)
    j = i.copy()
    k = i.copy() * 2
    g = np.meshgrid(i, j, k, indexing="ij")
    g = np.vstack(list(map(np.ravel, g))).T

    for ac, bc, cc in g:
        scu = xyz + ac * a + bc * b + cc * c
        final.append(scu)

    totats = np.tile(ats, g.shape[0])
    final = np.asarray(final).reshape(-1, 3)
    struct = np.c_[ totats, final ]

    with open("azulene555.xyz", "w") as f:
        np.savetxt(f, struct, fmt="%-3s %12.6f %12.6f %12.6f", header="%d\n" % struct.shape[0], comments='')
