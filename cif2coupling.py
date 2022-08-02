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
#---------------------------------------------------------------
# DISTANCE return the distance "dist" between 2 objects with
def distance(x1,y1,z1,x2,y2,z2):
    """
    Return the distance 'dist' between 2 objects with coordinates
    x1,y1,z1;x2,y2,z2   
    ----------
    INPUT: 
    x1,y1,z1;x2,y2,z2 : float
    ----------
    RETURNS
    dist: float
    """
    dist=np.sqrt( ((x1-x2)**2) + ((y1-y2)**2) +((z1-z2)**2) )

    return dist


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

disorder_tag=False
with open(cif_file,'r') as oldfile:
	for line in oldfile:
        	if  "disorder" in line:
            		disorder_tag=True
# If there is disorder then use ASE workaround 
if disorder_tag:
	full_struct = ase.io.read(cif_file, store_tags=True)
	tags = full_struct.info.copy()
	is_valid_atom = [dg in ('.', 1) for dg in tags["_atom_site_disorder_group"]]
	for tag in ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z','_atom_site_type_symbol', '_atom_site_label'):
		if tag not in tags: continue
		tags[tag] = [el for is_ok, el in zip(is_valid_atom, tags[tag]) if is_ok]
	atoms_ase = ase.io.cif.tags2atoms(tags)
# If I don't have disorder, it simply reads the file as it is
else:
	atoms_ase=ase.io.read(cif_file)

  
##############################################################################
# OLD VERSION
 
xyz_file=os.path.splitext(cif_file)[0]+'.xyz'

# Read with ASE
disorder_tag=False
with open(cif_file,'r') as oldfile:
	for line in oldfile:
        	if  "disorder" in line:
            		disorder_tag=True

if disorder_tag:
	full_struct = ase.io.read(cif_file, store_tags=True)
	tags = full_struct.info.copy()
	is_valid_atom = [dg in ('.', 1) for dg in tags["_atom_site_disorder_group"]]
	for tag in ('_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z','_atom_site_type_symbol', '_atom_site_label'):
		if tag not in tags: continue
		tags[tag] = [el for is_ok, el in zip(is_valid_atom, tags[tag]) if is_ok]
	atoms_ase = ase.io.cif.tags2atoms(tags)

else:
	atoms_ase=ase.io.read(cif_file)

Zmolecules=1
if 'string' in locals():
	Zmolecules=int(re.findall(r"[-+]?\d*\.\d+|\d+", string)[0])

# REad XYZ
element = []; x=[]; y=[]; z=[];
with open(xyz_file,'r') as f:
     natoms=(int(f.readline()))/Zmolecules#UPDATE natoms (maybe here it's not necessary?)
     f.next()
     for line in f:
      elem,xval,yval,zval = line.split()
      element.append(str(elem)); x.append(float(xval)); y.append(float(yval)); z.append(float(zval))
# SUpercell
element_0=[]; x_0=[]; y_0=[]; z_0=[];
for i in range(natoms*Zmolecules):
        element_0.append(element[i])
        x_0.append(x[i])
        y_0.append(y[i])
        z_0.append(z[i])

# For now I work with a supercell number [1,1,1].
num_cell=[1,1,1]

for i in range(-num_cell[0],num_cell[0]+1):
 for j in range(-num_cell[1],num_cell[1]+1):
  for k in range(-num_cell[2],num_cell[2]+1):
    if i!=0 or j!=0 or k!=0:  
     for ii in range(natoms*Zmolecules):
        element.append(element_0[ii])
        xval=x_0[ii]+i*float(atoms_ase.cell[0,0])
        yval=y_0[ii]+i*float(atoms_ase.cell[0,1])
        zval=z_0[ii]+i*float(atoms_ase.cell[0,2])
        xval=xval+j*float(atoms_ase.cell[1,0])
        yval=yval+j*float(atoms_ase.cell[1,1])
        zval=zval+j*float(atoms_ase.cell[1,2])
        xval=xval+k*float(atoms_ase.cell[2,0])
        yval=yval+k*float(atoms_ase.cell[2,1])
        zval=zval+k*float(atoms_ase.cell[2,2])
        x.append(xval)
        y.append(yval)
        z.append(zval)

natoms_tot=len(x)


  
# ID
xcell=[];ycell=[];zcell=[];ID_zmol=[]
for ii in range(Zmolecules):
        ID_zmol.append(ii)
        xcell.append(0)
        ycell.append(0)
        zcell.append(0)
for i in range(-num_cell[0],num_cell[0]+1):
 for j in range(-num_cell[1],num_cell[1]+1):
  for k in range(-num_cell[2],num_cell[2]+1):
    if i!=0 or j!=0 or k!=0: 
     for ii in range(Zmolecules):
        ID_zmol.append(ii)
        xcell.append(i)
        ycell.append(j)
        zcell.append(k)
        
#       IDENTIFICATION OF NEAREST NEIGHBOURS

element_base=[]; x_base=[]; y_base=[]; z_base=[];
for i in range(natoms):
        element_base.append(element[i])
        x_base.append(x[i])
        y_base.append(y[i])
        z_base.append(z[i])

# Now I check all atoms closer than 1.55 Angstrom (C-X bond)  
# So I have the number of atoms bonded to each Carbon atom
threshold_atoms=1.55
neighbours=[]
for i in range(natoms):
 k=0
 for j in range(natoms):
        if i!=j:
                dist=distance(x_base[i],y_base[i],z_base[i],x_base[j],y_base[j],z_base[j])
                if dist<threshold_atoms:
                        k=k+1
 neighbours.append(k)

conj_list=[]
for i in range(natoms):
        if neighbours[i]<=3 and element_base[i]=='C':
                conj_list.append(i)
natoms_conj=len(conj_list)

red_neig_list=[]
for i in range(Zmolecules):
        for j in range(i+1,nmol_tot):
                near_neigh=False
                for ii in range(natoms_conj):
                        for jj in range(natoms_conj):
                                ind1=conj_list[ii] + i*natoms
                                ind2=conj_list[jj] + j*natoms
                                dist=distance(x[ind1],y[ind1],z[ind1],x[ind2],y[ind2],z[ind2])
                                if dist<threshold:
                                        near_neigh=True
                if near_neigh:
                        red_neig_list.append([i,j,ID_zmol[j],xcell[j],ycell[j],zcell[j]])  


#"NOTE: lists also symmetry-equivalent couples"

num_red_neig=len(red_neig_list)

#
#       ELIMINATING REDUNDANCIES i.e. identical couples
# Up to now checking the distances between centres of mass

Mass_list=[]
for i in range(natoms):
        if element[i]=="C" or element[i]=="c":
                Mass_list.append(12.0107)
        if element[i]=="H" or element[i]=="h":
                Mass_list.append(1.00794)
        if element[i]=="N" or element[i]=="n":
                Mass_list.append(14.0067)
        if element[i]=="O" or element[i]=="o":
                Mass_list.append(15.9994)
        if element[i]=="Si" or element[i]=="si":
                Mass_list.append(28.0855)
        if element[i]=="S" or element[i]=="s":
                Mass_list.append(32.0600)
        if element[i]=="F" or element[i]=="f":
                Mass_list.append(18.9984)
        if element[i]=="Ge" or element[i]=="ge":
                Mass_list.append(72.64)


Mass_tot=0
for i in range(natoms):
        Mass_tot=Mass_tot+Mass_list[i]
with open("foo.txt", "w") as f: 
        for i in range(natoms):
                f.write(str(Mass_list[i])+' '+str(element[i])+'\n' )

Mass_centers=[]
for i in range(nmol_tot):
        xval=0.0
        yval=0.0
        zval=0.0
        for j in range(natoms):
                xval=xval+Mass_list[j]*x[i*(natoms)+j]
                yval=yval+Mass_list[j]*y[i*(natoms)+j]
                zval=zval+Mass_list[j]*z[i*(natoms)+j]

        Mass_centers.append([xval/Mass_tot,yval/Mass_tot,zval/Mass_tot])


# Now Not-redundant
diff_MC=[]
for i in range(num_red_neig):
        j=red_neig_list[i][0]
        k=red_neig_list[i][1]
        dist=distance(Mass_centers[j][0],Mass_centers[j][1],Mass_centers[j][2],Mass_centers[k][0],Mass_centers[k][1],Mass_centers[k][2])        
        diff_MC.append(dist)


neighbour_list=[]
for i in reversed(range(num_red_neig)):
        near_neigh=True
        for j in range(i):
                if np.absolute(diff_MC[i]-diff_MC[j])<(np.absolute(diff_MC[i])/1000):
                        near_neigh=False
        if near_neigh:
                neighbour_list.append(red_neig_list[i])


num_neig=len(neighbour_list)

   

        
        
   

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
