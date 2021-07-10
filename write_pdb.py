#!/usr/bin/env python
import sys
from read_pdb import PDB

class PDBio(PDB):

    template = "{head:6s}{serial:5d}  {name:<3}{altLoc:1s}{resName:3s} {chainID:1s}{resSeq:4d}{iCode:1s}   {x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{tempFactor:6.2f}          {element:>2s}{charge:2s}\n"

    def write_pdb(self, filename, chainID=None, atoms=None, MODEL=[None, 1]):
        if not atoms: atoms = self.atoms
        f = open(filename, 'w')
        for atom in atoms:
            atom.head = 'ATOM'
            if chainID:
                if atom.chainID != chainID: continue
            if not atom.MODEL in MODEL: continue
            f.write(self.template.format(**atom.__dict__))
        f.write('TER')
        f.close()

if __name__ == '__main__':
    pdb = PDBio(sys.argv[1])
    filename = './test.pdb'
    chainID = 'A'
    pdb.write_pdb(filename, chainID)