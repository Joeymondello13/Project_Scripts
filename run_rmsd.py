import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
if __name__ == '__main__':
    for i in range(len(os.listdir("pdb_files1"))):
        for j in range(len(os.listdir("pdb_files2"))):
            os.system("python RMSD.py" + ' ' +  str(i) + ' ' + str(j))
    #os.system("python RMSD.py" + ' ' + str(1) + ' ' + str(1))