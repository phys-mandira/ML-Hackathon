import numpy as np


base_name = 'fileName_'
extn = '.xyz'
width = 4
size = 10
input_file='dataset_train.csv'
if __name__ == '__main__':
    with open(input_file) as fin:
        for i in range(1, 5001):
            file_name = base_name + (str(i)).rjust(width, '0') + extn
            #print(file_name)
            fin2 =  open(file_name, 'a')
            prop = np.genfromtxt(fin, dtype = float, comments=None, max_rows=1, usecols= [1,2,3,4])
            fin2.write('\n{} {} {} {}'.format(prop[0], prop[1], prop[2], prop[3]))
