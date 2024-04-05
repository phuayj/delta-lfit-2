import itertools
import os
import sys

import click

from delta_lfit_2.delta_lfit_2 import DeltaLFIT2
from delta_lfit_2.logic import (
    GENERATE_PYTHON,
    GENERATE_READABLE,
    get_max_rule_len,
    get_rule_mappings,
    generate_transition_steps,
    length,
    human_program,
    index_to_prog,
    subsumes,
)

import numpy as np

import torch

from tqdm.auto import tqdm


def convert_state_to_numbers(state):
    return sum([s * (2**i) for i, s in enumerate(state)])


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def find_subsets(s, n):
    return list(itertools.combinations(s, n))


@click.command()
@click.option('--config', default='./configs/train_var3.py', help='Config file')
@click.option('--test_prog', default='./programs/n3a.py', help='Test program file')
@click.option('--noise', default=0., help='Noise ratio', type=float)
@click.option('--observable', default=1., help='Observable state percentage', type=float)
@click.option('--threshold', default=0.4, help='Rule prediction threshold', type=float)
@click.option('--print_transitions', is_flag=True, default=False, help='Print out the original and predicted state transitions')
def main(config, test_prog, noise, observable, threshold, print_transitions):
    conf = {}
    with open(config, 'r') as f:
        exec(f.read(), None, conf)

    if not os.path.exists(conf['model_file']):
        print(f'Model file "{conf["model_file"]}" not found!')
        sys.exit(1)
        return

    num_vars, delays = conf['num_vars'], conf['delays']

    model = DeltaLFIT2(
        num_vars, delays,
        conf['set_transformer_encoder_config'],
        conf['set_transformer_decoder_config'],
        conf['program_predictor_config'],
    )
    model = model.to(conf['device'])
    model.load_state_dict(torch.load(conf['model_file']))

    prog = {}
    with open(test_prog, 'r') as f:
        exec(f.read(), None, prog)

    # Generate the transitions from the test program
    observed_transitions = []
    func_str = human_program(prog['prog'], '', prog['num_vars']*prog['delays'], GENERATE_PYTHON)
    globals_scope = {
        'generate_transition_steps': generate_transition_steps,
        'seq_len': 2,
    }
    local = {}
    exec(func_str, globals_scope, local)
    for starting_state in range(2 ** (prog['num_vars']*prog['delays'])):
        local['starting_state'] = starting_state
        local['num_vars'] = prog['num_vars']*prog['delays']
        exec(
            'transitions = generate_transition_steps(generate_transition, seq_len, num_vars, starting_state)',
            globals_scope,
            local,
        )
        transitions = local['transitions']
        observed_transitions.append(transitions)

    partial = int((2 ** (prog['num_vars']*prog['delays'])) * observable)
    observed_transitions = np.random.permutation(observed_transitions)[:partial]

    max_rule_len = get_max_rule_len(num_vars, delays)
    rule_len_masks = {}
    for rule_len in range(1, num_vars*delays+1):
        mask = torch.zeros(2 ** (num_vars*delays), max_rule_len, device=conf['device'], dtype=torch.bool)
        mask[:,:(length(rule_len, num_vars*delays) - length(rule_len-1, num_vars*delays))] = 1
        rule_len_masks[rule_len] = mask

    rule_mappings = get_rule_mappings(num_vars * delays)

    print('Generating subsets...')
    subsets = find_subsets(range(prog['num_vars']), num_vars) if num_vars < prog['num_vars'] else find_subsets(range(prog['num_vars']), prog['num_vars'])

    all_rules = []
    added_rules = set()
    pbar = tqdm(total=prog['num_vars']*len(subsets))
    for i in range(prog['num_vars']):
        for s in subsets:
            if i in s:
                s2 = list(s)
                idx = s2.index(i)
                s = s2[idx:] + s2[:idx]
            data = set()
            for transition in observed_transitions:
                new_t = (convert_state_to_numbers([1 if transition[0][x] else 0 for x in s]), convert_state_to_numbers([1 if transition[1][i] else 0]))
                data.add(new_t)
            data = torch.tensor([list(data)])
            in_data = torch.clone(data)
            batch_size = data.shape[0]
            max_state = torch.tensor((2 ** (num_vars*delays)) - 1, dtype=torch.long).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_size, in_data.shape[1], 2)
            num_vars_tensor = torch.tensor(num_vars*delays, dtype=torch.long).expand(batch_size)
            var_idx2 = torch.tensor([0]).unsqueeze(1)
            var_idx2 = var_idx2.unsqueeze(2).expand(batch_size, in_data.shape[1], 2)
            ls = (num_vars_tensor - 0).unsqueeze(1).unsqueeze(2).expand(batch_size, in_data.shape[1], 2)
            a = torch.bitwise_right_shift(in_data, var_idx2)
            b = torch.bitwise_and(a, max_state)
            c = torch.bitwise_left_shift(in_data, ls)
            d = torch.bitwise_and(c, max_state)
            in_data = torch.bitwise_or(b, d)
            var_idx = torch.ones_like(data, device=conf['device'])
            in_data[:,:,1] = (in_data[:,:,1] & 1) > 0
            inp = in_data[:,:,0] + (2 ** (num_vars*delays)) * in_data[:,:,1]
            in_data = inp.to(conf['device'])
            rule_len = torch.arange(num_vars*delays+1).to(conf['device'], dtype=torch.long)
            in_data = in_data.expand(num_vars*delays+1, data.shape[1])
            in_data = inp.to(conf['device'])
            rule_len = torch.arange(num_vars*delays+1).to(conf['device'], dtype=torch.long)
            in_data = in_data.expand(num_vars*delays+1, data.shape[1])
            preds_out = model(in_data, rule_len, rule_len_is_diff=True)
            for rule_len in range(1, num_vars*delays+1):
                preds_out[rule_len][:,:-1][~rule_len_masks[rule_len][:1,:]] = 0
            preds = []
            for j in range(0, num_vars*delays+1):
                if j > 0:
                    t = torch.sigmoid(torch.cat((preds_out[j,0,:length(j, num_vars*delays)-length(j-1, num_vars*delays)], preds_out[j,0,-1].unsqueeze(0))))
                else:
                    t = torch.sigmoid(torch.cat((preds_out[j,0,:1], preds_out[j,0,-1].unsqueeze(0))))
                if t[-1] > 0.5:
                    t = t[:-1]
                    t = torch.zeros_like(t, device=conf['device'])
                else:
                    t = t[:-1]
                preds.append(t)
            preds = torch.cat(preds)
            preds = preds[rule_mappings[0]]
            preds = (preds >= threshold).bool()
            pred_prog_rules = []
            for r_i, r in enumerate(preds):
                if r:
                    new_r = index_to_prog(r_i, num_vars*delays)
                    new_r[1] = to_tuple(new_r[1])
                    conflicts = []
                    new_body = set(new_r[1])
                    var_in_rules = set([b[1][0] for b in new_r[1]])
                    to_add = set([to_tuple(new_r)])
                    while len(to_add) > 0:
                        new_add = set()
                        for new_r in to_add:
                            do_append = True
                            do_add = True
                            post_pred_prog_rules = []
                            for a in pred_prog_rules:
                                if new_r[0] == a[0] and subsumes(a[1], new_r[1]):
                                    post_pred_prog_rules.append(a)
                                    do_append = False
                                    do_add = False
                                elif new_r[0] == a[0] and subsumes(new_r[1], a[1]):
                                    do_append = True
                                elif new_r[0] == a[0] and len(new_body.intersection(set(a[1]))) > 0:
                                    intersected_vars = set([v[1][0] for v in new_body.intersection(set(a[1]))])
                                    a_var_in_rules = set([b[1][0] for b in a[1]])
                                    new_left = var_in_rules - intersected_vars
                                    a_left = a_var_in_rules - intersected_vars
                                    if new_left == a_left:
                                        a_rule = (a[0], to_tuple(new_body.intersection(set(a[1]))))
                                        new_add.add(a_rule)
                                        do_append = False
                                        do_add = False
                                    else:
                                        post_pred_prog_rules.append(a)
                                else:
                                    post_pred_prog_rules.append(a)
                            if do_append:
                                post_pred_prog_rules.append(new_r)
                            elif do_add:
                                new_add.add(new_r)
                            pred_prog_rules = post_pred_prog_rules
                        to_add = new_add
            for r in pred_prog_rules:
                new_r = (i, [])
                new_rule = f'{i}_'
                for b in r[1]:
                    if b[1][0] >= len(s):
                        continue
                    new_r[1].append((b[0], (s[b[1][0]], 1)))
                    new_rule = f'{new_rule}_{b[0]}_{s[b[1][0]]}'
                if new_rule not in added_rules:
                    all_rules.append(new_r)
                    added_rules.add(new_rule)
            pbar.update(1)
    pbar.close()

    print('Predicted:')
    new_p = []
    for r in all_rules:
        new_r = []
        for l in r[1]:
            new_r.append((l[0], (l[1][0] % prog['num_vars'], (l[1][0] // prog['num_vars'])+1)))
        new_p.append((r[0], new_r))
    print(human_program(new_p, '', prog['num_vars'], GENERATE_READABLE))

    print('Truth:')
    new_p = []
    for r in prog['prog']:
        new_r = []
        for l in r[1]:
            new_r.append((l[0], (l[1][0] % prog['num_vars'], (l[1][0] // prog['num_vars'])+1)))
        new_p.append((r[0], new_r))
    print(human_program(new_p, '', prog['num_vars'], GENERATE_READABLE))

    func_str = human_program(prog['prog'], '', prog['num_vars'], GENERATE_PYTHON)
    globals_scope = {
        'generate_transition_steps': generate_transition_steps,
        'seq_len': 2,
    }
    local = {}
    local['num_vars'] = prog['num_vars']
    exec(func_str, globals_scope, local)
    data = []
    for starting_state in range(2 ** (prog['num_vars']*prog['delays'])):
        local['starting_state'] = starting_state
        exec(
            'transitions = generate_transition_steps(generate_transition, seq_len, num_vars, starting_state)',
            globals_scope,
            local,
        )
        transitions = local['transitions']
        data.append([[1 if x else 0 for x in transitions[0]], [1 if x else 0 for x in transitions[1]]])
    truth_data = np.array(data)

    func_str = human_program(all_rules, '', prog['num_vars'], GENERATE_PYTHON)
    globals_scope = {
        'generate_transition_steps': generate_transition_steps,
        'seq_len': 2,
    }
    local = {}
    local['num_vars'] = prog['num_vars']
    exec(func_str, globals_scope, local)
    data = []
    for starting_state in range(2 ** (prog['num_vars']*prog['delays'])):
        local['starting_state'] = starting_state
        exec(
            'transitions = generate_transition_steps(generate_transition, seq_len, num_vars, starting_state)',
            globals_scope,
            local,
        )
        transitions = local['transitions']
        data.append([[1 if x else 0 for x in transitions[0]], [1 if x else 0 for x in transitions[1]]])
    pred_data = np.array(data)

    if print_transitions:
        for i, s in enumerate(truth_data):
            print(s[0], s[1], pred_data[i][1])

    print('MSE:', (np.square(pred_data[:,1,:prog['num_vars']] - truth_data[:,1,:prog['num_vars']]).mean()))


if __name__ == '__main__':
    main()
