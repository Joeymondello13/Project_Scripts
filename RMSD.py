#!/usr/bin/env python
import copy
import numpy as np
import os
import sys
from write_pdb import PDBio

def fit_rms(ref_c,c):
    # move geometric center to the origin
    ref_trans = np.average(ref_c, axis=0)
    ref_c = ref_c - ref_trans
    c_trans = np.average(c, axis=0)
    c = c - c_trans

    # covariance matrix
    C = np.dot(c.T, ref_c)

    # Singular Value Decomposition
    (r1, s, r2) = np.linalg.svd(C)

    # compute sign (remove mirroring)
    if np.linalg.det(C) < 0:
        r2[2,:] *= -1.0
    U = np.dot(r1, r2)
    return (c_trans, U, ref_trans)

class RMSDcalculator:
    def __init__(self, atoms1, atoms2, name=None):
        xyz1 = self.get_xyz(atoms1, name=name)
        xyz2 = self.get_xyz(atoms2, name=name)
        self.set_rmsd(xyz1, xyz2)

    def get_name(self, atoms, name=None):
        names = []
        for atom in atoms:
            if name:
                if atom.name != name: continue
                names.append[atom.name]
        return np.array(names)

    def get_xyz(self, atoms, name=None):
        xyz = []
        for atom in atoms:
            if name:
                if atom.name != name: continue
            xyz.append([atom.x, atom.y, atom.z])
        return np.array(xyz)

    def set_rmsd(self, c1, c2):
        self.rmsd = 0.0
        self.c_trans, self.U, self.ref_trans = fit_rms(c1, c2)
        new_c2 = np.dot(c2 - self.c_trans, self.U) + self.ref_trans
        self.rmsd = np.sqrt( np.average( np.sum( ( c1 - new_c2 )**2, axis=1 ) ) )

    def get_aligned_coord(self, atoms, name=None):
        new_c2 = copy.deepcopy(atoms)
        for atom in new_c2:
            atom.x, atom.y, atom.z = np.dot(np.array([atom.x, atom.y, atom.z]) - self.c_trans, self.U) + self.ref_trans
        return new_c2


if __name__ == '__main__':
    pdb_files1 = os.listdir('pdb_files1')
    pdb_files2 = os.listdir('pdb_files2')
    result_file = open('results.txt', 'a')
    matrix_file = open('matrix_file.txt', 'a')
    newline = False
    index1 = sys.argv[1]
    index2 = sys.argv[2]
    pdbf1 = 'pdb_files1/' + pdb_files1[int(index1)]; pdbf2 = 'pdb_files2/' + pdb_files2[int(index2)]
    pdb1 = PDBio(pdbf1); pdb2 = PDBio(pdbf2)
    atoms1 = pdb1.get_atoms(to_dict=False); atoms2 = pdb2.get_atoms(to_dict=False)
    RMSDcalculator = RMSDcalculator(atoms1, atoms2, name=None)
    rmsd = RMSDcalculator.rmsd
    n = RMSDcalculator.get_name(atoms1)
    print(n)
    print(rmsd)
    #for i in range(len(pdb_files1)):
        #for j in range(len(pdb_files2)):
    print(str(pdb_files1[int(index1)]))
    result_file.write(
        "\nPDB1: " + str(pdb_files1[int(index1)]) + "\nPDB2: " + str(pdb_files2[int(index2)]) + "\nRMSD: " + str(rmsd) + "\n"
                      "------------------------------------------------------------------------------\n")
    if int(index2) < 12:
        matrix_file.write(str(rmsd) + ', ')
    else:
        matrix_file.write(str(rmsd) + '\n')
    result_file.close()
    matrix_file.close()