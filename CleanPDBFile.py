#!/usr/bin/env python
# coding: utf-8

# In[98]:


import Bio
import re
from Bio.PDB import PDBList
'''Selecting structures from PDB'''
pdbl = PDBList()
#4 digit PDB code of structure
accession_code='6XM4'
#Name of folder to download PDB file
pdir = 'PDB_files'

file_format="pdb"
#This function downloads the PDB file from the databank
pdbl.retrieve_pdb_file(accession_code, obsolete=False, pdir=pdir, file_format=file_format, overwrite=False)
#Stores pdb file into list
with open("C:\\Users\Mondello\\PDB_files\\pdb6xm4.ent", "r") as file:
    pdb_file=file.read()
file.close()    
#Finds all lines that does NOT have HETATM (non-protein atoms)
x=re.findall('^(?!^HETATM).*',pdb_file,flags=re.MULTILINE)
#Writes all lines except non-protein lines from PDB file
with open("C:\\Users\Mondello\\PDB_files\\pdb6xm4_clean.pdb", "w") as file2:
    for i in x:
        if not i.isspace():
            file2.write(i)
            file2.write("\n")
file2.close()            


# In[ ]:





# In[ ]:




