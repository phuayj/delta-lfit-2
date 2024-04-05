import os

import click

from delta_lfit_2.dataloader import TransitionDataset, transform_batch
from delta_lfit_2.delta_lfit_2 import DeltaLFIT2

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm


@click.command()
@click.option('--config', default='./configs/train_var3.py', help='Config file')
def main(config):
    conf = {}
    with open(config, 'r') as f:
        exec(f.read(), None, conf)

    num_vars, delays = conf['num_vars'], conf['delays']

    train_size = round(conf['total_data_size']*conf['train_size'])
    test_size = round(conf['total_data_size']*conf['test_size'])
    val_size = conf['total_data_size'] - train_size - test_size

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
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=conf['val_batch_size'],
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=conf['test_batch_size'],
        shuffle=True,
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

    model = model.to(conf['device'])

    start_epoch = 0
    if os.path.exists(conf['checkpoint_file']):
        checkpoint = torch.load(conf['checkpoint_file'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        rule_lengths_stats = checkpoint['rule_lengths_stats']

    criterion = nn.BCEWithLogitsLoss()

    pbar = None
    try:
        for epoch in range(start_epoch, conf['epochs']):
            model.train()
            pbar = tqdm(total=int(len(train_data) / conf['train_batch_size']) + 1)
            losses = []
            for batch, batched_data in enumerate(train_dataloader):
                optimizer.zero_grad()
                trained_batch = False
                rand = np.random.random_sample()
                if rand > conf['sparse_probability']:
                    checks = conf['sparse_lengths']
                else:
                    checks = np.random.permutation(range(0, num_vars*delays+1))
                data = targets = rule_lens = None
                for rule_len in checks:
                    data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, conf['device'], num_vars, delays)
                    zeros = torch.sum(targets[:,-1]).item()
                    if zeros > batched_data[0].shape[0] * conf['sparse_limit'] and rand < conf['sparse_probability']:
                        continue
                    trained_batch = True
                    rule_lengths_stats[f'rl{rule_len}'] += 1
                    break
                if not trained_batch:
                    rule_len = np.random.choice(conf['sparse_lengths'])
                    data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, conf['device'], num_vars, delays)
                    rule_lengths_stats[f'rl{rule_len}'] += 1
                predictions = model(data, rule_lens)
                loss = criterion(predictions.squeeze(), targets)

                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                if batch > 0 and batch % conf['checkpoint_step'] == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'rule_lengths_stats': rule_lengths_stats,
                    }, conf['checkpoint_file'])
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), **rule_lengths_stats)
            pbar.close()
            pbar = None

            avg_loss = np.mean(losses)
            print(f'Epoch {epoch}: train loss {avg_loss:.3f}')

            model.eval()
            losses = []
            for batched_data in test_dataloader:
                rule_len = np.random.randint(num_vars*delays) + 1  # 0 is trivial
                data, targets, rule_lens, var_idx = transform_batch(batched_data, rule_len, conf['device'], num_vars, delays)
                predictions = model(data, rule_lens)
                loss = criterion(predictions.squeeze(), targets)
                losses.append(loss.item())

            avg_loss = np.mean(losses)
            print(f'Epoch {epoch}: test loss {avg_loss:.3f}')
    except KeyboardInterrupt:
        if pbar:
            pbar.close()
        print('Aborted')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'rule_lengths_stats': rule_lengths_stats,
    }, conf['checkpoint_file'])
    torch.save(model.state_dict(), conf['model_file'])


if __name__ == '__main__':
    main()
