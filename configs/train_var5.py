# Only for use with train_accelerator.py
import torch


num_vars = 5
delays = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = 'dataset_var5.hdf5'

transition_pct = 0.75

total_data_size = int(2e9)+int(1e7)
val_size = 0.15
test_size = 0.1
train_size = 0.75

train_batch_size = 512
train_num_workers = 8
val_batch_size = 16
test_batch_size = 16

set_transformer_encoder_config = {
    'dim_in': 1024,
    'dim_out': 1024,
    'layers': 5,
    'num_heads': 8,
    'use_isab': True,
    'ln': True,
    'dropout': 0.2,
}

set_transformer_decoder_config = {
    'dim_in': 1024,
    'dim_hidden': 1024,
    'dim_out': 1024,
    'num_outputs': 1,
    'layers': 3,
    'num_heads': 8,
    'ln': True,
    'dropout': 0.2,
}

program_predictor_config = {
    'dim_in': 1024,
    'dim_hidden': 1024,
    'layers': 4,
    'dropout': 0.2,
    'ln': True,
}

optimizer = 'AdamW'
optimizer_config = {
    'lr': 1e-5,
    'weight_decay': 1e-4,
}
epochs = 100

checkpoint_file = 'model_var5.ckpt'

sparse_probability = 0.85
sparse_lengths = [0, 1]
sparse_limit = 0.95

model_file = 'model_var5.weights'

checkpoint_step = 5000

model_dir = './model_var5/'
