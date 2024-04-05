import hashlib
import pickle
import time

import click

from delta_lfit_2.logic import(
    program_from_transitions,
    convert_program_to_one_hot,
)

import h5py

import numpy as np

import redis


def generate_single_datapoint(num_vars, delays):
    state_transitions = {}
    prog_hash = b''
    for i in range(2**(num_vars*delays)):
        state_transitions[i] = np.random.randint(0, 2**num_vars)
        prog_hash += i.to_bytes(1, 'little') + state_transitions[i].to_bytes(1, 'little')  # Only for less than 8 variables
    new_p = program_from_transitions(state_transitions, num_vars, delays)
    one_hot_prog = convert_program_to_one_hot(new_p, num_vars*delays)[:num_vars]
    one_hot_prog = one_hot_prog.reshape(-1)
    return state_transitions, one_hot_prog, prog_hash


def generate_dataset(seed, can_stop, redis_host, redis_port, num_vars, delays):
    np.random.seed(seed)
    r = redis.Redis(host=redis_host, port=redis_port)
    try:
        while not can_stop.is_set():
            if r.llen(f'generate-queue:{num_vars}:{delays}') > 1e7:
                time.sleep(5)
                continue
            state_transitions, one_hot_prog, prog_hash = generate_single_datapoint(num_vars, delays)
            if not r.sadd(f'seen-hashes:{num_vars}:{delays}', prog_hash):
                continue
            r.rpush(f'generate-queue:{num_vars}:{delays}', pickle.dumps((state_transitions, one_hot_prog, prog_hash)))
    except KeyboardInterrupt:
        return
