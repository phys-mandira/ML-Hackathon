import numpy as np

#data_path = "/home/dglab/koushik/schnet/Arpan_hack/source/"
data_path = "/home/dglab/mandira/Hackathon/Arpan_hack/source/"
base_name = "fileName_"
extn = ".xyz"
width = 4
x = 5000 # count  of data file upto we make the input 


if __name__ == "__main__":

    positions = []
    symbols = []
    prop1, prop2, prop3, prop4 = [], [], [], []

    #for i in range(1,4501):
    for i in range(4501, x+1):
        file_name = data_path + base_name + (str(i)).rjust(width, '0') + extn

        symbols_key = np.genfromtxt(file_name, dtype = str, skip_header= 2, skip_footer= 1, usecols=[0] )
        #print(symbols_key)

        temp = []
        for elem in symbols_key:
            temp.append(elem)
        symbols.append(temp)
        #print("symbols", symbols)
        temp = np.genfromtxt(file_name, skip_header= 2, skip_footer= 1, usecols=[1,2,3] )
        positions.append(temp)
        #print(temp)
        #print("positions", positions)


        tmp1, tmp2, tmp3, tmp4 = [], [], [], []
        atoms_num = int(np.genfromtxt(file_name, dtype = int, skip_header= 0, max_rows=1))
        p1, p2, p3, p4  = np.genfromtxt(file_name, dtype = float, skip_header= atoms_num+2, max_rows=1, usecols= [0,1,2,3])

        #print(p1,p2,p3,p4)
        tmp1.append(p1)
        prop1.append(tmp1)

        tmp2.append(p2)
        prop2.append(tmp2)

        tmp3.append(p3)
        prop3.append(tmp3)

        tmp4.append(p4)
        prop4.append(tmp4)

        #print(i)     
    # save information in numpy zip array, dtype object for asymetirc data type
    #np.savez( 'data_file_upto_'+str(4500)+".npz", Z=np.array(symbols, dtype=object), R=np.array(positions, dtype=object), prop1=np.array(prop1), prop2=np.array(prop2), prop3=np.array(prop3), prop4=np.array(prop4))
    np.savez( 'test_data_file_'+str(500)+".npz", Z=np.array(symbols, dtype=object), R=np.array(positions, dtype=object), prop1=np.array(prop1), prop2=np.array(prop2), prop3=np.array(prop3), prop4=np.array(prop4))
