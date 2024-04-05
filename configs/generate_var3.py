# Number of variables
num_vars = 3

# Amount of delay, only tested with 1
delays = 1

# Name of the dataset file to save to
dataset_name = 'dataset_var3.hdf5'

# The chunk size of the hdf5 file
chunk_size = 5000

# The amount of datapoint to generate for train, test, validation
val_size = 1024
test_size = 1024
train_size = int(5e8)-val_size-test_size
