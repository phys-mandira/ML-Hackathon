import numpy as np
import subprocess as sp

input_file = 'dataset_train.csv'
base_name = "fileName_"
extn = ".xyz"
width = 4
size = 5000 # count  of data file upto we make the input




if __name__ == '__main__':
    with open(input_file) as fin:
        for i in range(5000):
            fout = open('temp.smi','w')
            geom = str(np.genfromtxt(fin, dtype = str, comments=None, max_rows=1, usecols= 0))

            fout.write(geom)
            #print(geom, i) 
            
            xyz_file_name = base_name + (str(i)).rjust(width, '0') + extn
            sp.call('obabel -i smi /home/dglab/koushik/schnet/Arpan_hack/source/temp.smi -o xyz -O '+ xyz_file_name +' --gen3d', shell=True)
