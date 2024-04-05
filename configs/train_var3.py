import torch


# Device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of variables
num_vars = 3

# Amount of delay, only tested with 1
delays = 1

# Filename of generated dataset
dataset_name = 'dataset_var3.hdf5'

# Percentage of transitions to use as training
transition_pct = 0.75

# The amount of datapoint contained for train, test, validation
total_data_size = 16775000
val_size = 0.045
test_size = 0.005
train_size = 0.95

# Batch size and number of workers for train, test, validation
train_batch_size = 256
train_num_workers = 8
val_batch_size = 16
test_batch_size = 16

# Configuration for SetTransformerEncoder
set_transformer_encoder_config = {
    'dim_in': 256,
    'dim_out': 256,
    'layers': 3,
    'num_heads': 8,
    'use_isab': True,
    'ln': True,
    'num_inds': 64,
    'dropout': 0.2,
}

# Configuration for SetTransformerDecoder
set_transformer_decoder_config = {
    'dim_in': 256,
    'dim_hidden': 256,
    'dim_out': 256,
    'num_outputs': 1,
    'layers': 1,
    'num_heads': 8,
    'ln': True,
    'dropout': 0.2,
}

# Configuration for ProgramPredictor
program_predictor_config = {
    'dim_in': 256,
    'dim_hidden': 1024,
    'layers': 3,
    'dropout': 0.2,
    'ln': True,
}

# Optimizer to use, SGD and AdamW available. SGD found to be better in practice
optimizer = 'SGD'
optimizer_config = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
}

# Number of epochs to train
epochs = 100

# Filename for checkpoint
checkpoint_file = 'model_var3.ckpt'

# Filename for final model weights
model_file = 'model_var3.weights'

# Checkpoint every n steps
checkpoint_step = 5000

# Parameters regarding sparsity in mini-batches
sparse_probability = 0.99
sparse_lengths = [0]
sparse_limit = 0.95
