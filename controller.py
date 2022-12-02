import kosh


"""
The purpose of the controller script is exactly as advertised:
to control the creation of python-based polycrystals by way of 
the command line

1. The researcher must utilise a built-in kosh loader to govern 
the loading of their data into memory. This requires cleaning of 
the data beforehand into a suitable format. More advanced users 
can implement a custom kosh loader and do everything in one shot.

    1a. This process will include defining metadata to describe 
    and relate raw datasets.
    
    NOTE: Remember a good launching point is a list of grain
    orientations. 

2. The researcher must define which transformers, and operators
to use on the data to process it and retrieve a dataset of interest.

3. The final product from this script will be a dataset, in a 
desired format, processed and ready for use in a machine learning
algorithm or otherwise. To promote repeatability and good record 
keeping the dataset is saved back into the kosh store under user-
defined metadata

Proposed heirarchy:

Dataset = SVE (list of grain orientations)
Ensemble = Ensemble of SVEs

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
    parser.add_argument('--num_SVEs', required=True)
    parser.add_argument('--operators', default=None, choices=[])
    parser.add_argument('--transformers' default=None, choices=[])
    parser.add_argument('--raw_metadata', required=True, 
                        help='metadata to describe the raw orientation data\
                              Format: {crystal_structure: ,\
                                       orientation_format: ,\
                                       grain_count: ,\
                                       data_source: }')
    parser.add_argument('--processed_metadata', required=True, 
                        help='metadata to describe processed data, \
                              Format: {origin: \ 
                                       scaling:}')

    #NOTE: origin would be the dataset id of origin

    store_path = args.store_path

    #
    # 1. Open a store, create datasets, and associate data
    #

    # open a new store
    store = kosh.connect(store_path)

    # create datasets with metadata
    # TODO: This could benefit from using enum for name in the future
    dataset_name = 'SVE_'
    for num in range(num_SVEs):
        ds = store.create(dataset_name+=num, metadata=raw_metadata)
        ds.associate()



if __name__ == '__main__':

    if rank == 0:
        main()
