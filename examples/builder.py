import kosh
import os

"""
The purpose of the controller script is exactly as advertised:
to control the creation of python-based polycrystals by way of 
the command line and a config.yaml file.

PyTexture takes in a set of orientations in any given format,
create a polycrystal object, and contains methods to manipulate
these objects. 
It utilizies Kosh to keep track of the data and do other things.

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
    parser.add_argument('--raw_data_path', required=True)
    parser.add_argument('--mime_type', required=True)
    parser.add_argument('--ds_metadata', required=True, 
                        help='metadata to describe the raw orientation data\
                              Format: {data_type: , \
                                      orientation_format: ,\
                                       grain_count: , }')
    parser.add_argument('--custom_loader', default=False)
    parser.add_argument('--loader_path', default='')
    parser.add_argument('--ens_name', required=True)
    parser.add_argument('--ens_metadata', default={})


    store_path = args.store_path
    raw_data_path = args.raw_data_path
    mime_type = args.mime_type
    ds_metadata = args.ds_metadata
    custom_loader = args.custom_loader
    loader_path = args.loader_path

    #
    # 1. Open a store, create dataset, and associate data
    #

    # Open a new store
    print(f'Opening kosh store at {store_path}')
    store = kosh.connect(store_path, delete_all_contents=True)

    # Check for custom loader
    store.add_loader(CustomLoader)

    print('Building Kosh dataset...')
    ens = store.create_ensemble(name=ensemble_name, metadata=ens_metadata)
    ds = ens.create(dataset_name, metadata=ds_metadata)
    ds.associate(raw_data_path, mime_type=mime_type)

    #
    # 2. From store, build Texture objects and save
    #

    print('Building Texture Object and Saving...')




if __name__ == '__main__':

    if rank == 0:
        ()
