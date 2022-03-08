# Written by AL & DP

# To insert
# Reading cif via API
# trim alkyl chain
# evaluate coupling
# diabatization
#

#####
#
#                VERSION 0.1
#
#        LAST MODIFIED 04/03/2022
#####

# MODULES
import numpy as np
import re
import subprocess 
import os
import glob
import ase.io
import ase.build
from ccdc import io 
import datetime 
import time 

#############################################
#               CONSTANTS (SI units)
#############################################

Temperature=300#Kelvin
hbar=1.0545718e-34#J s
KBoltzmann=1.38064852e-23#J/k
lightspeed=2.99792e8#m/si

#############################################
#               FUNCTIONS
#############################################
######################################################################################
print('{:>60}'.format(" !-----------------------------------------------------!"))
print('{:>60}'.format(" !                                                     !"))
print('{:>60}'.format(" !                                                     !"))
print('{:>60}'.format(" !                     Version 0.1                     !"))
print('{:>60}'.format(" !                                                     !"))
print('{:>60}'.format(" !                      Written by                     !"))
print('{:>60}'.format(" !          Alessandro Landi / Daniele Padula          !"))
print('{:>60}'.format(" !           alessandro.landi4869@gmail.com            !"))
print('{:>60}'.format(" !                                                     !"))
print('{:>60}'.format(" !                                                     !"))
print('{:>60}'.format(" !-----------------------------------------------------!"))

##################################################################################################
##################################################################################################
#
# INPUT SECTION!!!!
# CHANGE HERE for input parameters
#       
cif_file='Pn.cif'       # Name of the CIF file
threshold=5.00          # Angstrom to consider atoms near. Suggested=5.00
QM_software='Gaussian'  # "Gaussian"
noalkylchain=False      # True to substitute alkyl chain with a methyl
readGeometry=True       # If True, read geometry from an xyz file
                        # Useful if cif file is misunderstood by ASE or if we want to use optimized geometry
optimizeCell=False      # If True optimize unit cell, holding lattice constants fixed
evaluateCoupling=True   # If True evaluate Coupling.
allCoupling=False       # If False only evaluate HOMO/HOMO coupling - if True also LUMO-LUMO;HOMO-1;LUMO+1
evaluateOverlap=False   # If True evaluate Overlap in addition to coupling
##################################################################################################



##################################################################################################
# Evaluate the time of start
now = datetime.datetime.now(i)
start = time.time()# To evaluate the elapsed time
print("!-----------------------------------------------------!")
print("    Computations started at:")
print (str(now))
print("!-----------------------------------------------------!")



mol_file = str(os.path.splitext(os.path.basename(cif_file))[0])+".mol"
reader = io.CrystalReader(cif_file) # Leggo il cif
for c in reader:
    unit_cell_molecule = c.packing(((0, 0, 0), (0.9, 0.9, 0.9)))
    with io.CrystalWriter(mol_file) as crystal_writer:
            crystal_writer.write(unit_cell_molecule)
# Read as xyz

pass

##################################################################################################
#
# THE END
#
now = datetime.datetime.now()
print("!-----------------------------------------------------!")
print("    Computations finished at:")
print (str(now))
print("!-----------------------------------------------------!")
end = time.time()

print("!-----------------------------------------------------!")
print("    ELAPSED TIME:")
print(str(end - start))
print("!-----------------------------------------------------!")
#----------------------------------------------------------#
