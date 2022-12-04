import kosh
import os

"""
The purpose of the controller script is exactly as advertised:
to control the creation of python-based polycrystals by way of 
the command line

PyTexture takes in a set of orientations in any given format,
create a polycrystal object, and contains methods to manipulate
these objects. 
It utilizies Kosh to keep track of the data.

"""

try:

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

except AttributeError:

    rank = 0

def main():
    
    #
    # Command line interface 
    #

    parser = argparser.ArgumentParser()
    parser.add_argument('--store_path', required=True)
    parser.add_argument('--orientation_data_dir', required=True)
    parser.add_argument('--mime_type', required=True)
    parser.add_argument('--operators', default=None, choices=[])
    parser.add_argument('--transformers' default=None, choices=[])
    parser.add_argument('--ds_metadata', required=True, 
                        help='metadata to describe the raw orientation data\
                              Format: {data_type: , \
                                       orientation_format: ,\
                                       grain_count: , }')
    parser.add_argument('--ens_metadata', required=True, 
                        help='metadata to describe the raw orientation data\
                              Format: {crystal_structure: ,\
                                       data_source: }')
    parser.add_argument('--processed_metadata', required=True, 
                        help='metadata to describe processed data, \
                              Format: {origin: \ 
                                       scaling:}')
    parser.add_argument('--ensemble_name', required=True)

    #NOTE: origin would be the dataset id of origin

    store_path = args.store_path

    #
    # 1. Open a store, create datasets, and associate data
    #

    # open a new store
    store = kosh.connect(store_path)

    # create datasets with metadata and add to ensemble
    # TODO: This could benefit from using enum for name in the future
    
    dataset_name = 'SVE_'
    ens = store.create_ensemble(name=ensemble_name, metadata=ens_metadata)
    orientations = os.listdir(orientation_data_dir)

    print('Building Kosh Store...')
    for num, sve in enumerate(orientations):
        ds = store.create(dataset_name+=num, metadata=ds_metadata)
        ds.associate(sve, mime_type=mime_type)
        ens.associate(ds)
    
    #
    # 2. From store, build Texture objects and save
    #

    print('Building Texture Objects and Saving...')
    
    # Grab all relevant datasets
    datasets = list(ens.find_datasets(data_type='Orientations'))


    #
    # 3. Choose attributes to calculate and save 
    #



if __name__ == '__main__':

    if rank == 0:
        main()
