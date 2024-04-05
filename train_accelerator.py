import os

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

import click

from delta_lfit_2.dataloader import TransitionDataset
from delta_lfit_2.delta_lfit_2 import DeltaLFIT2
from delta_lfit_2.logic import get_max_rule_len, length

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm.auto import tqdm


logger = get_logger(__name__, log_level='DEBUG')


def transform_batch(data, rule_len, num_vars, delays):
    max_rule_len = get_max_rule_len(num_vars, delays)
    batch_size = data[0].shape[0]
    var_idx = data[2]
    if rule_len > 0:
        rules_idx = torch.arange(length(rule_len-1, num_vars*delays), length(rule_len, num_vars*delays)).unsqueeze(0)
    else:
        rules_idx = torch.arange(0, length(rule_len, num_vars*delays)).unsqueeze(0)
    rules_idx = rules_idx.expand(batch_size, rules_idx.shape[1])
    target = data[1]
    empty_pred = data[3]
    rules_idxes = torch.gather(target, 1, rules_idx.to(target.device))[:,:max_rule_len]
    if rules_idxes.shape[1] < max_rule_len:
        rules_idxes = torch.cat((rules_idxes, torch.zeros(batch_size, max_rule_len - rules_idxes.shape[1], device=rules_idxes.device)), 1)
    rule_idxes = rules_idxes.bool()
    prog = torch.zeros(batch_size, max_rule_len, dtype=torch.float, device=rule_idxes.device)
    prog[rule_idxes] = 1
    prog = torch.cat((prog, empty_pred[:,rule_len:rule_len+1].to(prog.device)), dim=1)
    inp = data[0]
    targets = prog
    rule_lens = torch.tensor(rule_len).unsqueeze(0).expand(batch_size)
    return inp[:,:-1], targets, rule_lens, var_idx


@click.command()
@click.option('--config', default='./configs/train_var5.py', help='Config file')
@click.option('--resume', is_flag=True, default=False, help='Resume from checkpoint')
def main(config, resume):
    conf = {}
    with open(config, 'r') as f:
        exec(f.read(), None, conf)

    num_vars, delays = conf['num_vars'], conf['delays']

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Make sure distributed models are initialized to the same values
    torch.manual_seed(42)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    shards_per_process = conf['total_data_size'] // world_size
    assert shards_per_process > 0, 'Not enough shards to split evenly'
    my_shard_start = rank * shards_per_process
    my_shard_length = min(conf['total_data_size'], (rank+1) * shards_per_process) - my_shard_start

    assert conf['train_size'] + conf['val_size'] + conf['test_size'] == 1
    #train_size = round(conf['train_size']*my_shard_length)
    #test_size = round(conf['test_size']*my_shard_length)
    train_size = round(conf['train_size']*conf['total_data_size'])
    test_size = round(conf['test_size']*conf['total_data_size'])
    val_size = conf['total_data_size'] - train_size - test_size
    assert train_size + test_size + val_size == conf['total_data_size'], f'{train_size} + {test_size} + {val_size} = {train_size + test_size + val_size} != {conf["total_data_size"]}'

    train_data = TransitionDataset(
        conf['dataset_name'],
        num_vars, delays,
        0, train_size,
        conf.get('transition_pct', 1.0),
    )
    test_data = TransitionDataset(
        conf['dataset_name'],
        num_vars, delays,
        train_size, test_size,
        conf.get('transition_pct', 1.0),
    )
    val_data = TransitionDataset(
        conf['dataset_name'],
        num_vars, delays,
        train_size + test_size, val_size,
        conf.get('transition_pct', 1.0),
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=conf['train_batch_size'],
        shuffle=True,
        num_workers=conf['train_num_workers'],
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=conf['val_batch_size'],
        num_workers=2,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=conf['test_batch_size'],
    )

    model = DeltaLFIT2(
        num_vars, delays,
        conf['set_transformer_encoder_config'],
        conf['set_transformer_decoder_config'],
        conf['program_predictor_config'],
    )
    if conf['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            **conf['optimizer_config'],
        )
    elif conf['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            **conf['optimizer_config'],
        )

    rule_lengths_stats = {
        f'rl{i}': 0
        for i in range(num_vars*delays+1)
    }

    model = model.to(accelerator.device)

    start_epoch = 0

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,
    )

    if resume:
        accelerator.load_state(conf['model_dir'])

    criterion = nn.BCEWithLogitsLoss()

    pbar = None
    try:
        for epoch in range(start_epoch, conf['epochs']):
            model.train()
            pbar = tqdm(total=int(len(train_data) / conf['train_batch_size']) + 1, disable=not accelerator.is_local_main_process)
            losses = []
            for batch, batched_data in enumerate(train_dataloader):
                optimizer.zero_grad()
                trained_batch = False
                rand = np.random.random_sample()
                if rand > conf['sparse_probability']:
                    checks = np.random.permutation(conf['sparse_lengths'])
                else:
                    checks = np.random.permutation(range(0, num_vars*delays+1))
                for rule_len in checks:
                    data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, num_vars, delays)
                    zeros = torch.sum(targets[:,-1]).item()
                    if zeros > batched_data[0].shape[0] * conf['sparse_limit'] and rand < conf['sparse_probability']:
                        continue
                    trained_batch = True
                    rule_lengths_stats[f'rl{rule_len}'] += 1
                    break
                if not trained_batch:
                    rule_len = np.random.choice(conf['sparse_lengths'])
                    data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, num_vars, delays)
                    rule_lengths_stats[f'rl{rule_len}'] += 1
                predictions = model(data, rule_lens)
                loss = criterion(predictions.squeeze(), targets)

                accelerator.backward(loss)
                optimizer.step()

                losses.append(loss.item())
                if batch > 0 and batch % conf['checkpoint_step'] == 0:
                    '''
                    TODO
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'rule_lengths_stats': rule_lengths_stats,
                    }, conf['checkpoint_file'])
                    '''
                    accelerator.save_state(conf['model_dir'])
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), **rule_lengths_stats)

            accelerator.wait_for_everyone()
            pbar.close()
            pbar = None

            avg_loss = np.mean(losses)
            print(f'Epoch {epoch}: train loss {avg_loss:.3f}')

            model.eval()
            losses = []
            for batched_data in val_dataloader:
                rule_len = np.random.randint(num_vars*delays) + 1  # 0 is trivial
                data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, num_vars, delays)
                predictions = model(data, rule_lens)
                loss = criterion(predictions.squeeze(), targets.to(predictions.device))
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f'Epoch {epoch}: test loss {avg_loss:.3f}')
    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        print('Aborted')

    '''
    TODO
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'rule_lengths_stats': rule_lengths_stats,
    }, conf['checkpoint_file'])
    '''
    accelerator.wait_for_everyone()
    accelerator.save_model(model, conf['model_dir'])


if __name__ == '__main__':
    main()
