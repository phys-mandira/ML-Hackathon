import numpy as np
import sys
import os


from ase import Atoms
from schnetpack.data import ASEAtomsData


try:
    data = np.load('test_data_file_500.npz', mmap_mode='r',  allow_pickle=True)
except FileNotFoundError:
     sys.exit( "Given npz file not exist on the directory. Redo with valid file.")

#for k in data.files:
    #print ("k", k)

if __name__ == "__main__":

    # reaiding from npz file
    atoms_list = []
    property_list = []
    for symbols, positions, prop1, prop2, prop3, prop4 in zip(data['Z'], data['R'], data['prop1'], data['prop2'], data['prop3'], data['prop4']):
        #print("symbols", symbols)
        #print("positions", positions)
        ats = Atoms(positions=positions, symbols=symbols)
        #print("ats", ats)
        atoms_list.append(ats)
        properties = {"prop1": prop1, "prop2": prop2, "prop3": prop3, "prop4": prop4}
        #print("properties", properties)
        property_list.append(properties)
    #print("property_list", property_list)
    #print("atoms_list", atoms_list)
    #print(atoms_list[9].symbols)

    # Creating .db file
    file_name = 'upto_500.db'
    try:
        os.remove(file_name)
    except FileNotFoundError:
        pass

    data_set = ASEAtomsData.create(
            file_name,
            distance_unit = 'Ang',
            property_unit_dict = {'prop1': 'Ha',  'prop2': 'Ha',  'prop3': 'Ha', 'prop4': 'Ha'}
    )

    data_set.add_systems(property_list, atoms_list)
    print('Number of reference calculations:', len(data_set))
    print('Available properties:')
    for p in data_set.available_properties:
        print('-', p)
    #print(data_set)

    print('Properties of molecule with id 0:')
    example = data_set[0]
    for k, v in example.items():
        print('-', k, ':', v, ':', v.shape)

