# Script to generate training/val/test datasets.
# Requires Redis to run.
# Optionally run generate_dataset_worker.py on other machines to accelerate.
import os
import time
import sys
from multiprocessing import Event, Process, Manager

import click

from delta_lfit_2.dataloader import (
    get_dtypes,
)
from delta_lfit_2.generator import (
    generate_dataset,
)

import h5py

import numpy as np

import pickle

import redis

from tqdm import tqdm


@click.command()
@click.option('--config', default='./configs/generate_var3.py', help='Config file')
@click.option('--cpus', default=None, help='Number of CPUs to consume', type=int)
@click.option('--hide_progress', is_flag=True, default=False, help='Hide progressbar')
@click.option('--redis_host', default='localhost', help='Hostname of Redis server')
@click.option('--redis_port', default=6379, help='Port of Redis server')
@click.option('--resume', is_flag=True, default=False, help='Resume from existing file')
def main(config, cpus, hide_progress, redis_host, redis_port, resume):
    conf = {}
    with open(config, 'r') as f:
        exec(f.read(), None, conf)

    if os.path.exists(conf['dataset_name']) and not resume:
        print(f'Dataset "{conf["dataset_name"]}" already exists! Rename or remove file to continue.')
        sys.exit(1)
        return

    if cpus is None:
        cpus = os.cpu_count()

    show_progress = not hide_progress

    num_vars, delays = conf['num_vars'], conf['delays']

    dtypes = get_dtypes(num_vars, delays)

    f = h5py.File(conf['dataset_name'], 'a')
    if not resume or 'logic_programs' not in f:
        f.create_dataset(
            'logic_programs',
            (0,),
            dtype=dtypes['logic_program'],
            chunks=True,
            maxshape=(None,),
            compression='lzf',
        )
    if not resume or 'transitions' not in f:
        f.create_dataset(
            'transitions',
            (0, 2**num_vars*delays),
            dtype=dtypes['transition'],
            chunks=True,
            maxshape=(None, 2**num_vars*delays),
            compression='lzf',
        )
    logic_program_dataset = f['logic_programs']
    transition_dataset = f['transitions']

    generated = len(logic_program_dataset)

    with Manager() as manager:
        can_stop = Event()
        processes = []
        for j in range(cpus - 1):  # 1 CPU for the writer
            seed = int(j + time.time() % 10000)
            p = Process(target=generate_dataset, args=(
                seed, can_stop, redis_host, redis_port,
                num_vars, delays,
            ))
            p.start()
            processes.append(p)

        r = redis.Redis(host=redis_host, port=redis_port)

        if show_progress:
            pbar = tqdm(total=conf['train_size'] + conf['val_size'] + conf['test_size'], initial=generated)
        else:
            pbar = None

        errors = 0
        current_lp_rows = []
        current_t_rows = []

        try:
            while generated < conf['train_size'] + conf['val_size'] + conf['test_size']:
                data = r.blpop(f'generate-queue:{num_vars}:{delays}')
                state_transitions, one_hot_prog, prog_hash = pickle.loads(data[1])
                current_lp_rows.append((prog_hash, one_hot_prog))
                current_t_rows.append(list(state_transitions.items()))
                generated += 1
                if len(current_lp_rows) >= conf['chunk_size']:
                    if pbar:
                        pbar.set_postfix(size=r.llen(f'generate-queue:{num_vars}:{delays}'), errors=errors)
                    logic_program_dataset.resize((logic_program_dataset.shape[0] + len(current_lp_rows),))
                    transition_dataset.resize((transition_dataset.shape[0] + len(current_t_rows), transition_dataset.shape[1]))
                    p = np.array(current_lp_rows, dtype=dtypes['logic_program'])
                    logic_program_dataset[-len(current_lp_rows):] = p
                    t = np.array(current_t_rows, dtype=dtypes['transition'])
                    transition_dataset[-len(current_t_rows):,:] = t
                    if pbar:
                        pbar.update(len(current_lp_rows))
                    current_lp_rows = []
                    current_t_rows = []

            if len(current_lp_rows) > 0:
                if pbar:
                    pbar.set_postfix(size=r.llen(f'generate-queue:{num_vars}:{delays}'), errors=errors)
                logic_program_dataset.resize((logic_program_dataset.shape[0] + len(current_lp_rows),))
                transition_dataset.resize((transition_dataset.shape[0] + len(current_t_rows), transition_dataset.shape[1]))
                p = np.array(current_lp_rows, dtype=logic_program_dtype)
                logic_program_dataset[-len(current_lp_rows):] = p
                t = np.array(current_t_rows, dtype=transition_dtype)
                transition_dataset[-len(current_t_rows):,:] = t
                if pbar:
                    pbar.update(len(current_lp_rows))
        except KeyboardInterrupt:
            print('Aborted')
            if len(current_lp_rows) > 0:
                if pbar:
                    pbar.set_postfix(size=r.llen(f'generate-queue:{num_vars}:{delays}'), errors=errors)
                logic_program_dataset.resize((logic_program_dataset.shape[0] + len(current_lp_rows),))
                transition_dataset.resize((transition_dataset.shape[0] + len(current_t_rows), transition_dataset.shape[1]))
                p = np.array(current_lp_rows, dtype=logic_program_dtype)
                logic_program_dataset[-len(current_lp_rows):] = p
                t = np.array(current_t_rows, dtype=transition_dtype)
                transition_dataset[-len(current_t_rows):,:] = t
                if pbar:
                    pbar.update(len(current_lp_rows))
        finally:
            can_stop.set()
            for p in processes:
                p.join()
            if pbar:
                pbar.close()
            f.close()

    print('Done')


if __name__ == '__main__':
    main()
