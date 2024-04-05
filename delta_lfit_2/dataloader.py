from delta_lfit_2.logic import length, get_rule_mappings, get_max_rule_len

import h5py

import numpy as np

import torch
from torch.utils.data import Dataset


def get_dtypes(num_vars, delays):
    return {
        'logic_program': np.dtype([('hash', 'S16'), ('one_hot_prog', np.uint8, (num_vars*delays * length(num_vars*delays, num_vars*delays),))]),
        'transition': np.dtype([('state1', 'i4'), ('state2', 'i4')]),
    }


def transform_batch(data, rule_len, device, num_vars, delays):
    max_rule_len = get_max_rule_len(num_vars, delays)
    batch_size = data[0].shape[0]
    var_idx = data[2]
    if rule_len > 0:
        rules_idx = torch.arange(length(rule_len-1, num_vars*delays), length(rule_len, num_vars*delays), device=device).unsqueeze(0)
    else:
        rules_idx = torch.arange(0, length(rule_len, num_vars*delays), device=device).unsqueeze(0)
    rules_idx = rules_idx.expand(batch_size, rules_idx.shape[1])
    target = data[1].to(device=device)
    empty_pred = data[3].to(device)
    rules_idxes = torch.gather(target, 1, rules_idx)[:,:max_rule_len]
    if rules_idxes.shape[1] < max_rule_len:
        rules_idxes = torch.cat((rules_idxes, torch.zeros(batch_size, max_rule_len - rules_idxes.shape[1], device=device)), 1)
    rule_idxes = rules_idxes.bool()
    prog = torch.zeros(batch_size, max_rule_len, dtype=torch.float, device=device)
    prog[rule_idxes] = 1
    prog = torch.cat((prog, empty_pred[:,rule_len:rule_len+1]), dim=1)
    inp = data[0]
    targets = prog
    rule_lens = torch.tensor(rule_len).unsqueeze(0).expand(batch_size)
    return inp[:,:-1].to(device), targets.to(device), rule_lens.to(device), var_idx.to(device)


class TransitionDataset(Dataset):
    def __init__(self, dataset_name, num_vars, delays, base_idx, size, transition_pct=1.0):
        super(TransitionDataset, self).__init__()
        self.dataset_name = dataset_name
        self.transition_pct = transition_pct
        self.num_vars = num_vars*delays
        self.rule_mappings = get_rule_mappings(num_vars*delays)
        self.len = None
        with h5py.File(self.dataset_name, 'r', swmr=True) as f:
            dataset_size = len(f['logic_programs'])
        if base_idx + size > dataset_size:
            raise Exception(f'Base index {base_idx} with size {size} is larger than dataset size {dataset_size}')
        self.base_idx = base_idx
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            raise NotImplementedError

        idx += self.base_idx
        with h5py.File(self.dataset_name, 'r', swmr=True) as f:
            prog = f['logic_programs'][idx]
            data = f['transitions'][idx,:]

        data = np.hstack(data)
        data = torch.tensor(data.tolist())  # Somehow data is np.void
        data = data[:int(2**self.num_vars * self.transition_pct),:]
        prog = torch.tensor(prog[1], dtype=torch.float)

        # Select a random variable to train for
        var_idx = np.random.randint(self.num_vars)

        # Slice the program to the chosen variable
        prog = prog[var_idx * (3 ** self.num_vars) : (var_idx+1) * (3 ** self.num_vars)]

        # Remap the rules to the chosen variable
        new_idxes = []
        target = torch.zeros_like(prog)
        empty_pred = [0] * (self.num_vars+1)
        for rule_len in range(self.num_vars+1):
            rules = torch.nonzero(prog[(length(rule_len-1, self.num_vars) if rule_len > 0 else 0):length(rule_len, self.num_vars)], as_tuple=False)
            new_prog = []
            for idx in rules:
                new_prog.append(self.rule_mappings[var_idx][(length(rule_len-1, self.num_vars) if rule_len > 0 else 0) + idx])
            new_idxes = torch.tensor(new_prog, dtype=torch.long)
            target[new_idxes] = 1
            if len(new_idxes) == 0:
                empty_pred[rule_len] = 1

        # Now let's remap the state transitions
        max_state = torch.tensor((2 ** (self.num_vars)) - 1, dtype=torch.long).unsqueeze(0).unsqueeze(1).expand(data.shape[0], 2)
        num_vars_tensor = torch.tensor(self.num_vars, dtype=torch.long)
        var_idx2 = torch.tensor(var_idx).unsqueeze(0).expand(data.shape[0], 2)
        ls = (num_vars_tensor - var_idx).unsqueeze(0).expand(data.shape[0], 2)
        data_state = torch.clone(data).detach()

        # Shift rotate the variables
        a = torch.bitwise_right_shift(data_state, var_idx2)
        b = torch.bitwise_and(a, max_state)
        c = torch.bitwise_left_shift(data_state, ls)
        d = torch.bitwise_and(c, max_state)
        data_state = torch.bitwise_or(b, d)

        # Encode the state transitions w.r.t the chosen variable
        inp_state = torch.clone(data_state).detach()
        inp_state[:,1] = (inp_state[:,1] & 1) > 0
        inp = data_state[:,0] + (2 ** self.num_vars) * ((data_state[:,1] & 1) > 0)

        return (inp, target, var_idx, torch.tensor(empty_pred))
