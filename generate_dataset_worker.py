# Worker script, designed to run together with
# generate_dataset.py
import os
import time
from multiprocessing import Event, Process

import click

from delta_lfit_2.generator import (
    generate_dataset,
)


@click.command()
@click.option('--config', default='./configs/generate_var3.py', help='Config file')
@click.option('--cpus', default=None, help='Number of CPUs to consume', type=int)
@click.option('--redis_host', default='localhost', help='Hostname of Redis server')
@click.option('--redis_port', default=6379, help='Port of Redis server')
def main(config, cpus, redis_host, redis_port):
    conf = {}
    with open(config, 'r') as f:
        exec(f.read(), None, conf)

    if cpus is None:
        cpus = os.cpu_count()

    num_vars, delays = conf['num_vars'], conf['delays']

    can_stop = Event()
    processes = []
    for j in range(cpus):
        seed = int(j + time.time() % 10000)
        p = Process(target=generate_dataset, args=(
            seed, can_stop, redis_host, redis_port,
            num_vars, delays,
        ))
        p.start()
        processes.append(p)
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print('Aborted')
        can_stop.set()
        for p in processes:
            p.join()


if __name__ == '__main__':
    main()
