# Functions and helper code related to the logic part of dLFIT2.

import functools
import math

import numpy as np

from scipy.special import comb


GENERATE_READABLE = 'readable'
GENERATE_LP = 'lp'
GENERATE_PYTHON = 'py'

rule_mappings = None


def generate_rule(num_vars):
    rule_vars = np.random.binomial(2, 0.33, num_vars).tolist()
    body = [(0 if e == 1 else 1, (i, 1)) for i, e in enumerate(rule_vars) if e > 0]

    return body


def readable_rule(i, rule):
    return 'x{0}(T) :- {1}.'.format(
        i,
        ', '.join([
            '{0}x{1}(T-{2})'.format(
                'not ' if r[0] else '',
                r[1][0],
                r[1][1],
            )
            for r in rule
        ]),
    )


def human_program(program, ident, num_vars, generate_mode=GENERATE_LP):
    rules = []
    for rule_enc in program:
        i = rule_enc[0]
        rule = rule_enc[1]
        # Readable rules
        if generate_mode == GENERATE_READABLE:
            rules.append(readable_rule(i, rule))
        elif generate_mode == GENERATE_LP:
            for r in rule:
                # Generate false rules
                if r[0]:
                    rules.append(
                        'x{0}(0,T) :- x{1}(1,T-{2}).'.format(
                            i,
                            r[1][0],
                            r[1][1],
                        ),
                    )
                else:
                    rules.append(
                        'x{0}(0,T) :- x{1}(0,T-{2}).'.format(
                            i,
                            r[1][0],
                            r[1][1],
                        ),
                    )

            # Generate true rules
            rules.append('x{0}(1,T) :- {1}.'.format(
                i,
                ', '.join([
                    'x{0}({1},T-{2})'.format(
                        r[1][0],
                        0 if r[0] else 1,
                        r[1][1],
                    )
                    for r in rule
                ]),
            ))
        elif generate_mode == GENERATE_PYTHON:
            rules.append((i, ' and '.join([
                '{0}state[{1}]'.format(
                    'not ' if r[0] else '',
                    r[1][0],
                )
                for r in rule
            ])))
    if generate_mode == GENERATE_PYTHON:
        rule_vars = {}
        for r in rules:
            if r[0] not in rule_vars:
                rule_vars[r[0]] = []
            if r[1]:
                rule_vars[r[0]].append(r[1])
        for i in range(num_vars):
            if i not in rule_vars:
                rule_vars[i] = ['True']
        return (
            'def generate_transition{1}(state):\n'
            '    next_state = [False] * len(state)\n'
            '\n'
            '    {0}\n'
            '\n'
            '    return next_state\n'
            .format(
                '\n    '.join([
                    'next_state[{0}] = True if ({1}) else False'.format(
                        key,
                        ') or ('.join(value),
                    ) if len(value) > 0 else f'next_state[{key}] = True'
                    for key, value in rule_vars.items()
                ]),
                ident,
            ))
    elif generate_mode == GENERATE_READABLE:
        return '\n'.join(rules)
    return '{0}\n\n{1}'.format(
        '\n'.join([
            'VAR x{0} 0 1'.format(i) for i in range(num_vars)
        ]),
        '\n'.join(rules)
    )


def calc_minimality(curr_rules, rule_):
    if prog_subsumes(curr_rules, rule_):
        return False

    for r in curr_rules:
        if r[0] == rule_[0]:
            if subsumes(rule_[1], r[1]):
                return False
            r = set(r[1])
            rule = set(rule_[1])
            if len(r & rule) > 0:
                if len(r - rule) == 0 or len(rule - r) == 0:
                    return False
                if len(r - rule) == len(rule - r):
                    a = set([(1 if x[0] == 0 else 0, x[1]) for x in r - rule])
                    b = rule - r
                    if len(a ^ b) == 0:
                        return False
    return True


def generate_program(num_vars, minimal_only=False, init_rule_body=None):
    rules = []
    if init_rule_body:
        rules.append((
            init_rule_body[0],
            [(x[0], (x[1][0], x[1][1])) for x in init_rule_body[1]],
        ))
    is_minimal = True
    for i in np.random.permutation(num_vars):
        generate = True
        curr_num = 1 if init_rule_body and init_rule_body[0] == i else 0
        while generate:
            rule = generate_rule(num_vars)
            if len(rule) == 0:
                break
            rule.sort(key=lambda x: x[1][0])
            curr_minimality = calc_minimality(rules, (i, rule))
            if not minimal_only or curr_minimality:
                is_minimal = is_minimal and curr_minimality
                rules.append((i, rule))
                curr_num += 1
            rand = np.random.random_sample()
            stop = 0.5 ** curr_num
            if rand > stop:
                generate = False

    if len(rules) == 0:
        return generate_program(num_vars, minimal_only, init_rule_body)

    return rules, is_minimal


def encode_state(state):
    return ''.join([
        '1' if s else '0'
        for s in state
    ])


def generate_transition_steps(func, seq_len, num_vars, starting_state=0):
    state = [
        True if starting_state&(1<<i) != 0 else False
        for i in range(num_vars)
    ]
    result = [state]
    for step in range(seq_len-1):
        state = func(state)
        result.append(state)

    return result


def subsumes(rule_i, rule_j):
    rule_i = set(rule_i)
    rule_j = set(rule_j)
    if rule_i.issubset(rule_j):
        return True

    return False


def prog_subsumes(p, rule):
    for r in p:
        if r[0] == rule[0]:
            if subsumes(r[1], rule[1]):
                return True
    return False


def program_from_transitions(state_transitions, num_vars, delays):
    # The symbolic LFIT algorithm for minimal programs
    p = [(i, []) for i in range(num_vars)]
    for a in range(num_vars*delays):
        for e_i, e_j in state_transitions.items():
            if e_j & (1 << int(math.floor(a/delays))) == 0:
                rule = (int(math.floor(a/delays)), [
                    (0 if e_i & (1 << b) != 0 else 1,
                     (b, 1))
                    for b in range(num_vars*delays)
                ])
                conflicts = []
                for r in p:
                    if r[0] == rule[0] and subsumes(r[1], rule[1]):
                        conflicts.append(r)
                for r in conflicts:
                    p.remove(r)
                for r in conflicts:
                    for l in rule[1]:
                        con_l = [m for m in r[1] if m[1][0] == l[1][0]]
                        if len(con_l) == 0:
                            r_c = (r[0], [(x[0], (x[1][0], 1)) for x in r[1]])
                            r_c[1].append((0 if l[0] else 1, (l[1][0], 1)))
                            if not prog_subsumes(p, r_c):
                                p = [
                                    r_p for r_p in p
                                    if (
                                            r_c[0] == r_p[0] and not subsumes(r_c[1], r_p[1])
                                    ) or r_c[0] != r_p[0]
                                ]
                                p.append(r_c)
    new_p = []
    for r in p:
        new_r = []
        for l in r[1]:
            new_r.append((l[0], (l[1][0] // delays, (l[1][0] % delays)+1)))
        new_p.append((r[0], new_r))
    return new_p


def generate_all_transitions(func, num_vars):
    generated_transitions = []
    transitions = []
    for starting_state in range(2 ** num_vars):
        state = [
            True if starting_state&(1<<i) != 0 else False
            for i in range(num_vars)
        ]
        for step in range(100):
            curr_transition = [state]
            state = func(state)
            curr_transition.append(state)
            encoded_transition = '{0}->{1}'.format(
                encode_state(curr_transition[0]),
                encode_state(curr_transition[1]),
            )
            if encoded_transition not in generated_transitions:
                generated_transitions.append(encoded_transition)
                transitions.append(curr_transition)

    return transitions


def convert_program_to_one_hot(curr_program, num_vars):
    # Not strictly one-hot
    one_hot_program = np.zeros((num_vars, length(num_vars, num_vars)), dtype=np.uint8)
    for r in curr_program:
        one_hot_program[r[0], get_index(r[1], num_vars)] = 1

    return one_hot_program



@functools.lru_cache(maxsize=1024)
def length(rule_len, num_vars):
    if rule_len == 0:
        return 1
    if rule_len > num_vars:
        return 3 ** num_vars
    return int(comb(num_vars, rule_len) * (2 ** rule_len) + length(rule_len - 1, num_vars))


@functools.cache
def get_max_rule_len(num_vars, delays):
    max_rule_len = 0
    for i in range(1, num_vars*delays):  # 0 is always length 1
        max_rule_len = max(
            max_rule_len,
            length(i, num_vars*delays) - length(i-1, num_vars*delays),
        )
    return max_rule_len


def get_rule_mappings(num_vars):
    global rule_mappings
    if rule_mappings is None:
        rule_mappings = []
        for var_idx in range(num_vars):
            rule_mappings.append([])
            var_mapping = [0] * num_vars
            for i in range(num_vars):
                var_mapping[(i + var_idx) % num_vars] = i
            for rule_idx in range(length(num_vars, num_vars)):
                rule = index_to_rule(rule_idx, num_vars)
                for i, lit in enumerate(rule):
                    rule[i][1][0] = var_mapping[rule[i][1][0]]
                rule_mappings[var_idx].append(get_index(rule, num_vars))
            assert len(rule_mappings[var_idx]) == length(num_vars, num_vars)
    return rule_mappings


def test_length():
    assert length(2, 3) == 19
    assert length(3, 3) == 27
    assert length(4, 4) == 81
    assert length(4, 5) == 211
    assert length(5, 5) == 243


def get_pos_idx(curr_var, num_vars, i, rule_len, last_var):
    assert i < rule_len
    if i == 0 and curr_var == -1:
        return 0
    if i == rule_len - 1:
        return num_vars - last_var - 1
    result = 0
    for j in range(max(last_var + 1, i), curr_var + 1):
        result += get_pos_idx(num_vars - (rule_len - 1 - i), num_vars, i + 1, rule_len, j)
    return result


def get_real_pos_idx(rule_body, num_vars):
    pos_idx = 0
    last_var = -1
    for i in range(len(rule_body) - 1):
        if rule_body[i][1][0] > i and rule_body[i][1][0] > last_var + 1:
            pos_idx += get_pos_idx(rule_body[i][1][0] - 1, num_vars, i,
                                   len(rule_body), last_var)
        last_var = rule_body[i][1][0]
    pos_idx += (rule_body[-1][1][0] - last_var - 1)
    return pos_idx


def get_index(rule_body, num_vars):
    if len(rule_body) == 0:
        return 0

    result = length(len(rule_body)-1, num_vars)
    rule_body.sort(key=lambda x: x[1][0])

    first_var = rule_body[0][1][0]
    if len(rule_body) == 1:
        result += first_var
        if rule_body[0][0]:
            result += num_vars
    else:
        neg_pos = 0
        for i in range(len(rule_body)-1, -1, -1):
            neg_pos <<= 1
            if rule_body[i][0]:
                neg_pos |= 1
        result += neg_pos * comb(num_vars, len(rule_body), exact=True)
        result += get_real_pos_idx(rule_body, num_vars)

    assert result < length(num_vars, num_vars)  # Shouldn't happen!

    return result


def incr_rule(rule_body, num_vars, pos):
    rule_body[pos][1][0] += 1
    if rule_body[pos][1][0] >= num_vars:
        assert pos > 0  # Shouldn't happen!
        incr_rule(rule_body, num_vars, pos-1)
        rule_body[pos][1][0] = rule_body[pos-1][1][0]
        incr_rule(rule_body, num_vars, pos)


def index_to_rule(idx, num_vars):
    rule_len = 0
    rule_len_idx = 0
    if idx > 0:
        rule_len = 1
        rule_len_idx = 1
        for i in range(num_vars):
            if length(i, num_vars) - 1 >= idx:
                break
            rule_len_idx = length(i, num_vars)
            rule_len = i + 1
    rule_idx = idx - rule_len_idx
    combinations = int(comb(num_vars, rule_len))
    pos_neg_idx = rule_idx // combinations
    comb_idx = rule_idx % combinations
    rule_body = [[0, [x, 1]] for x in range(rule_len)]
    for _ in range(comb_idx):
        incr_rule(rule_body, num_vars, rule_len-1)
    for i in range(rule_len):
        if pos_neg_idx & (1 << i) != 0:
            rule_body[i][0] = 1
    return rule_body


def index_to_prog(idx, num_vars):
    head = idx // length(num_vars, num_vars)
    idx = idx % length(num_vars, num_vars)
    return [head, index_to_rule(idx, num_vars)]


def test_index_rules():
    indices = [
        ([], 3, 0),  # .
        ([(0, (0, 1))], 3, 1),  # x0.
        ([(0, (1, 1))], 3, 2),  # x1.
        ([(0, (2, 1))], 3, 3),  # x2.
        ([(1, (0, 1))], 3, 4),  # not x0.
        ([(1, (1, 1))], 3, 5),  # not x1.
        ([(1, (2, 1))], 3, 6),  # not x2.
        ([(0, (0, 1)), (0, (1, 1))], 3, 7),  # x0, x1.
        ([(0, (0, 1)), (0, (2, 1))], 3, 8),  # x0, x2.
        ([(0, (1, 1)), (0, (2, 1))], 3, 9),  # x1, x2.
        ([(1, (0, 1)), (0, (1, 1))], 3, 10),   # not x0, x1.
        ([(1, (0, 1)), (0, (2, 1))], 3, 11),  # not x0, x2.
        ([(1, (1, 1)), (0, (2, 1))], 3, 12),  # not x1, x2.
        ([(0, (0, 1)), (1, (1, 1))], 3, 13),  # x0, not x1.
        ([(0, (0, 1)), (1, (2, 1))], 3, 14),  # x0, not x2.
        ([(0, (1, 1)), (1, (2, 1))], 3, 15),  # x1, not x2.
        ([(1, (0, 1)), (1, (1, 1))], 3, 16),  # not x0, not x1.
        ([(1, (0, 1)), (1, (2, 1))], 3, 17),  # not x0, not x2.
        ([(1, (1, 1)), (1, (2, 1))], 3, 18),  # not x1, not x2.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1))], 3, 19),  # x0, x1, x2.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1))], 3, 20),  # not x0, x1, x2.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1))], 3, 21),  # x0, not x1, x2.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1))], 3, 22),  # not x0, not x1, x2.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1))], 3, 23),  # x0, x1, not x2.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1))], 3, 24),  # not x0, x1, not x2.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1))], 3, 25),  # x0, not x1, not x2.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1))], 3, 26),  # not x0, not x1, not x2.

        ([], 4, 0),  # .
        ([(0, (0, 1))], 4, 1),  # x0.
        ([(0, (1, 1))], 4, 2),  # x1.
        ([(0, (2, 1))], 4, 3),  # x2.
        ([(0, (3, 1))], 4, 4),  # x3.
        ([(1, (0, 1))], 4, 5),  # not x0.
        ([(1, (1, 1))], 4, 6),  # not x1.
        ([(1, (2, 1))], 4, 7),  # not x2.
        ([(1, (3, 1))], 4, 8),  # not x3.
        ([(0, (0, 1)), (0, (1, 1))], 4, 9),   # x0, x1.
        ([(0, (0, 1)), (0, (2, 1))], 4, 10),   # x0, x2.
        ([(0, (0, 1)), (0, (3, 1))], 4, 11),  # x0, x3.
        ([(0, (1, 1)), (0, (2, 1))], 4, 12),  # x1, x2.
        ([(0, (1, 1)), (0, (3, 1))], 4, 13),  # x1, x3.
        ([(0, (2, 1)), (0, (3, 1))], 4, 14),  # x2, x3.
        ([(1, (0, 1)), (0, (1, 1))], 4, 15),  # not x0, x1.
        ([(1, (0, 1)), (0, (2, 1))], 4, 16),  # not x0, x2.
        ([(1, (0, 1)), (0, (3, 1))], 4, 17),  # not x0, x3.
        ([(1, (1, 1)), (0, (2, 1))], 4, 18),  # not x1, x2.
        ([(1, (1, 1)), (0, (3, 1))], 4, 19),  # not x1, x3.
        ([(1, (2, 1)), (0, (3, 1))], 4, 20),  # not x2, x3.
        ([(0, (0, 1)), (1, (1, 1))], 4, 21),  # x0, not x1.
        ([(0, (0, 1)), (1, (2, 1))], 4, 22),  # x0, not x2.
        ([(0, (0, 1)), (1, (3, 1))], 4, 23),  # x0, not x3.
        ([(0, (1, 1)), (1, (2, 1))], 4, 24),  # x1, not x2.
        ([(0, (1, 1)), (1, (3, 1))], 4, 25),  # x1, not x3.
        ([(0, (2, 1)), (1, (3, 1))], 4, 26),  # x2, not x3.
        ([(1, (0, 1)), (1, (1, 1))], 4, 27),  # not x0, not x1.
        ([(1, (0, 1)), (1, (2, 1))], 4, 28),  # not x0, not x2.
        ([(1, (0, 1)), (1, (3, 1))], 4, 29),  # not x0, not x3.
        ([(1, (1, 1)), (1, (2, 1))], 4, 30),  # not x1, not x2.
        ([(1, (1, 1)), (1, (3, 1))], 4, 31),  # not x1, not x3.
        ([(1, (2, 1)), (1, (3, 1))], 4, 32),  # not x2, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1))], 4, 33),  # x0, x1, x2.
        ([(0, (0, 1)), (0, (1, 1)), (0, (3, 1))], 4, 34),  # x0, x1, x3.
        ([(0, (0, 1)), (0, (2, 1)), (0, (3, 1))], 4, 35),  # x0, x2, x3.
        ([(0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 36),  # x1, x2, x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1))], 4, 37),  # not x0, x1, x2.
        ([(1, (0, 1)), (0, (1, 1)), (0, (3, 1))], 4, 38),  # not x0, x1, x3.
        ([(1, (0, 1)), (0, (2, 1)), (0, (3, 1))], 4, 39),  # not x0, x2, x3.
        ([(1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 40),  # not x1, x2, x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1))], 4, 41),  # x0, not x1, x2.
        ([(0, (0, 1)), (1, (1, 1)), (0, (3, 1))], 4, 42),  # x0, not x1, x3.
        ([(0, (0, 1)), (1, (2, 1)), (0, (3, 1))], 4, 43),  # x0, not x2, x3.
        ([(0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 44),  # x1, not x2, x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1))], 4, 45),  # not x0, not x1, x2.
        ([(1, (0, 1)), (1, (1, 1)), (0, (3, 1))], 4, 46),  # not x0, not x1, x3.
        ([(1, (0, 1)), (1, (2, 1)), (0, (3, 1))], 4, 47),  # not x0, not x2, x3.
        ([(1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 48),  # not x1, not x2, x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1))], 4, 49),  # x0, x1, not x2.
        ([(0, (0, 1)), (0, (1, 1)), (1, (3, 1))], 4, 50),  # x0, x1, not x3.
        ([(0, (0, 1)), (0, (2, 1)), (1, (3, 1))], 4, 51),  # x0, x2, not x3.
        ([(0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 52),  # x1, x2, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1))], 4, 53),  # not x0, x1, not x2.
        ([(1, (0, 1)), (0, (1, 1)), (1, (3, 1))], 4, 54),  # not x0, x1, not x3.
        ([(1, (0, 1)), (0, (2, 1)), (1, (3, 1))], 4, 55),  # not x0, x2, not x3.
        ([(1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 56),  # not x1, x2, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1))], 4, 57),  # x0, not x1, not x2.
        ([(0, (0, 1)), (1, (1, 1)), (1, (3, 1))], 4, 58),  # x0, not x1, not x3.
        ([(0, (0, 1)), (1, (2, 1)), (1, (3, 1))], 4, 59),  # x0, not x2, not x3.
        ([(0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 60),  # x1, not x2, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1))], 4, 61),  # not x0, not x1, not x2.
        ([(1, (0, 1)), (1, (1, 1)), (1, (3, 1))], 4, 62),  # not x0, not x1, not x3.
        ([(1, (0, 1)), (1, (2, 1)), (1, (3, 1))], 4, 63),  # not x0, not x2, not x3.
        ([(1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 64),  # not x1, not x2, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 65),  # x0, x1, x2, x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 66),  # not x0, x1, x2, x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 67),  # x0, not x1, x2, x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 4, 68),  # not x0, not x1, x2, x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 69),  # x0, x1, not x2, x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 70),  # not x0, x1, not x2, x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 71),  # x0, not x1, not x2, x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 4, 72),  # not x0, not x1, not x2, x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 73),  # x0, x1, x2, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 74),  # not x0, x1, x2, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 75),  # x0, not x1, x2, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 4, 76),  # not x0, not x1, x2, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 77),  # x0, x1, not x2, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 78),  # not x0, x1, not x2, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 79),  # x0, not x1, not x2, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 4, 80),  # not x0, not x1, not x2, not x3.

        ([], 5, 0),  # .
        ([(0, (0, 1))], 5, 1),   # x0.
        ([(0, (1, 1))], 5, 2),   # x1.
        ([(0, (2, 1))], 5, 3),   # x2.
        ([(0, (3, 1))], 5, 4),   # x3.
        ([(0, (4, 1))], 5, 5),   # x4.
        ([(1, (0, 1))], 5, 6),   # not x0.
        ([(1, (1, 1))], 5, 7),   # not x1.
        ([(1, (2, 1))], 5, 8),   # not x2.
        ([(1, (3, 1))], 5, 9),   # not x3.
        ([(1, (4, 1))], 5, 10),  # not x4.
        ([(0, (0, 1)), (0, (1, 1))], 5, 11),  # x0, x1.
        ([(0, (0, 1)), (0, (2, 1))], 5, 12),  # x0, x2.
        ([(0, (0, 1)), (0, (3, 1))], 5, 13),  # x0, x3.
        ([(0, (0, 1)), (0, (4, 1))], 5, 14),  # x0, x4.
        ([(0, (1, 1)), (0, (2, 1))], 5, 15),  # x1, x2.
        ([(0, (1, 1)), (0, (3, 1))], 5, 16),  # x1, x3.
        ([(0, (1, 1)), (0, (4, 1))], 5, 17),  # x1, x4.
        ([(0, (2, 1)), (0, (3, 1))], 5, 18),  # x2, x3.
        ([(0, (2, 1)), (0, (4, 1))], 5, 19),  # x2, x4.
        ([(0, (3, 1)), (0, (4, 1))], 5, 20),  # x3, x4.
        ([(1, (0, 1)), (0, (1, 1))], 5, 21),  # not x0, x1.
        ([(1, (0, 1)), (0, (2, 1))], 5, 22),  # not x0, x2.
        ([(1, (0, 1)), (0, (3, 1))], 5, 23),  # not x0, x3.
        ([(1, (0, 1)), (0, (4, 1))], 5, 24),  # not x0, x4.
        ([(1, (1, 1)), (0, (2, 1))], 5, 25),  # not x1, x2.
        ([(1, (1, 1)), (0, (3, 1))], 5, 26),  # not x1, x3.
        ([(1, (1, 1)), (0, (4, 1))], 5, 27),  # not x1, x4.
        ([(1, (2, 1)), (0, (3, 1))], 5, 28),  # not x2, x3.
        ([(1, (2, 1)), (0, (4, 1))], 5, 29),  # not x2, x4.
        ([(1, (3, 1)), (0, (4, 1))], 5, 30),  # not x3, x4.
        ([(0, (0, 1)), (1, (1, 1))], 5, 31),  # x0, not x1.
        ([(0, (0, 1)), (1, (2, 1))], 5, 32),  # x0, not x2.
        ([(0, (0, 1)), (1, (3, 1))], 5, 33),  # x0, not x3.
        ([(0, (0, 1)), (1, (4, 1))], 5, 34),  # x0, not x4.
        ([(0, (1, 1)), (1, (2, 1))], 5, 35),  # x1, not x2.
        ([(0, (1, 1)), (1, (3, 1))], 5, 36),  # x1, not x3.
        ([(0, (1, 1)), (1, (4, 1))], 5, 37),  # x1, not x4.
        ([(0, (2, 1)), (1, (3, 1))], 5, 38),  # x2, not x3.
        ([(0, (2, 1)), (1, (4, 1))], 5, 39),  # x2, not x4.
        ([(0, (3, 1)), (1, (4, 1))], 5, 40),  # x3, not x4.
        ([(1, (0, 1)), (1, (1, 1))], 5, 41),  # not x0, not x1.
        ([(1, (0, 1)), (1, (2, 1))], 5, 42),  # not x0, not x2.
        ([(1, (0, 1)), (1, (3, 1))], 5, 43),  # not x0, not x3.
        ([(1, (0, 1)), (1, (4, 1))], 5, 44),  # not x0, not x4.
        ([(1, (1, 1)), (1, (2, 1))], 5, 45),  # not x1, not x2.
        ([(1, (1, 1)), (1, (3, 1))], 5, 46),  # not x1, not x3.
        ([(1, (1, 1)), (1, (4, 1))], 5, 47),  # not x1, not x4.
        ([(1, (2, 1)), (1, (3, 1))], 5, 48),  # not x2, not x3.
        ([(1, (2, 1)), (1, (4, 1))], 5, 49),  # not x2, not x4.
        ([(1, (3, 1)), (1, (4, 1))], 5, 50),  # not x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1))], 5, 51),  # x0, x1, x2.
        ([(0, (0, 1)), (0, (1, 1)), (0, (3, 1))], 5, 52),  # x0, x1, x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (4, 1))], 5, 53),  # x0, x1, x4.
        ([(0, (0, 1)), (0, (2, 1)), (0, (3, 1))], 5, 54),  # x0, x2, x3.
        ([(0, (0, 1)), (0, (2, 1)), (0, (4, 1))], 5, 55),  # x0, x2, x4.
        ([(0, (0, 1)), (0, (3, 1)), (0, (4, 1))], 5, 56),  # x0, x3, x4.
        ([(0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 57),  # x1, x2, x3.
        ([(0, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 58),  # x1, x2, x4.
        ([(0, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 59),  # x1, x3, x4.
        ([(0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 60),  # x2, x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1))], 5, 61),  # not x0, x1, x2.
        ([(1, (0, 1)), (0, (1, 1)), (0, (3, 1))], 5, 62),  # not x0, x1, x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (4, 1))], 5, 63),  # not x0, x1, x4.
        ([(1, (0, 1)), (0, (2, 1)), (0, (3, 1))], 5, 64),  # not x0, x2, x3.
        ([(1, (0, 1)), (0, (2, 1)), (0, (4, 1))], 5, 65),  # not x0, x2, x4.
        ([(1, (0, 1)), (0, (3, 1)), (0, (4, 1))], 5, 66),  # not x0, x3, x4.
        ([(1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 67),  # not x1, x2, x3.
        ([(1, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 68),  # not x1, x2, x4.
        ([(1, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 69),  # not x1, x3, x4.
        ([(1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 70),  # not x2, x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1))], 5, 71),  # x0, not x1, x2.
        ([(0, (0, 1)), (1, (1, 1)), (0, (3, 1))], 5, 72),  # x0, not x1, x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (4, 1))], 5, 73),  # x0, not x1, x4.
        ([(0, (0, 1)), (1, (2, 1)), (0, (3, 1))], 5, 74),  # x0, not x2, x3.
        ([(0, (0, 1)), (1, (2, 1)), (0, (4, 1))], 5, 75),  # x0, not x2, x4.
        ([(0, (0, 1)), (1, (3, 1)), (0, (4, 1))], 5, 76),  # x0, not x3, x4.
        ([(0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 77),  # x1, not x2, x3.
        ([(0, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 78),  # x1, not x2, x4.
        ([(0, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 79),  # x1, not x3, x4.
        ([(0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 80),  # x2, not x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1))], 5, 81),  # not x0, not x1, x2.
        ([(1, (0, 1)), (1, (1, 1)), (0, (3, 1))], 5, 82),  # not x0, not x1, x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (4, 1))], 5, 83),  # not x0, not x1, x4.
        ([(1, (0, 1)), (1, (2, 1)), (0, (3, 1))], 5, 84),  # not x0, not x2, x3.
        ([(1, (0, 1)), (1, (2, 1)), (0, (4, 1))], 5, 85),  # not x0, not x2, x4.
        ([(1, (0, 1)), (1, (3, 1)), (0, (4, 1))], 5, 86),  # not x0, not x3, x4.
        ([(1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 87),  # not x1, not x2, x3.
        ([(1, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 88),  # not x1, not x2, x4.
        ([(1, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 89),  # not x1, not x3, x4.
        ([(1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 90),  # not x2, not x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1))], 5, 91),  # x0, x1, not x2.
        ([(0, (0, 1)), (0, (1, 1)), (1, (3, 1))], 5, 92),  # x0, x1, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (4, 1))], 5, 93),  # x0, x1, not x4.
        ([(0, (0, 1)), (0, (2, 1)), (1, (3, 1))], 5, 94),  # x0, x2, not x3.
        ([(0, (0, 1)), (0, (2, 1)), (1, (4, 1))], 5, 95),  # x0, x2, not x4.
        ([(0, (0, 1)), (0, (3, 1)), (1, (4, 1))], 5, 96),  # x0, x3, not x4.
        ([(0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 97),  # x1, x2, not x3.
        ([(0, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 98),  # x1, x2, not x4.
        ([(0, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 99),  # x1, x3, not x4.
        ([(0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 100),  # x2, x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1))], 5, 101),  # not x0, x1, not x2.
        ([(1, (0, 1)), (0, (1, 1)), (1, (3, 1))], 5, 102),  # not x0, x1, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (4, 1))], 5, 103),  # not x0, x1, not x4.
        ([(1, (0, 1)), (0, (2, 1)), (1, (3, 1))], 5, 104),  # not x0, x2, not x3.
        ([(1, (0, 1)), (0, (2, 1)), (1, (4, 1))], 5, 105),  # not x0, x2, not x4.
        ([(1, (0, 1)), (0, (3, 1)), (1, (4, 1))], 5, 106),  # not x0, x3, not x4.
        ([(1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 107),  # not x1, x2, not x3.
        ([(1, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 108),  # not x1, x2, not x4.
        ([(1, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 109),  # not x1, x3, not x4.
        ([(1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 110),  # not x2, x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1))], 5, 111),  # x0, not x1, not x2.
        ([(0, (0, 1)), (1, (1, 1)), (1, (3, 1))], 5, 112),  # x0, not x1, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (4, 1))], 5, 113),  # x0, not x1, not x4.
        ([(0, (0, 1)), (1, (2, 1)), (1, (3, 1))], 5, 114),  # x0, not x2, not x3.
        ([(0, (0, 1)), (1, (2, 1)), (1, (4, 1))], 5, 115),  # x0, not x2, not x4.
        ([(0, (0, 1)), (1, (3, 1)), (1, (4, 1))], 5, 116),  # x0, not x3, not x4.
        ([(0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 117),  # x1, not x2, not x3.
        ([(0, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 118),  # x1, not x2, not x4.
        ([(0, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 119),  # x1, not x3, not x4.
        ([(0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 120),  # x2, not x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1))], 5, 121),  # not x0, not x1, not x2.
        ([(1, (0, 1)), (1, (1, 1)), (1, (3, 1))], 5, 122),  # not x0, not x1, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (4, 1))], 5, 123),  # not x0, not x1, not x4.
        ([(1, (0, 1)), (1, (2, 1)), (1, (3, 1))], 5, 124),  # not x0, not x2, not x3.
        ([(1, (0, 1)), (1, (2, 1)), (1, (4, 1))], 5, 125),  # not x0, not x2, not x4.
        ([(1, (0, 1)), (1, (3, 1)), (1, (4, 1))], 5, 126),  # not x0, not x3, not x4.
        ([(1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 127),  # not x1, not x2, not x3.
        ([(1, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 128),  # not x1, not x2, not x4.
        ([(1, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 129),  # not x1, not x3, not x4.
        ([(1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 130),  # not x2, not x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 131),  # x0, x1, x2, x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 132),  # x0, x1, x2, x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 133),  # x0, x1, x3, x4.
        ([(0, (0, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 134),  # x0, x2, x3, x4.
        ([(0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 135),  # x1, x2, x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 136),  # not x0, x1, x2, x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 137),  # not x0, x1, x2, x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 138),  # not x0, x1, x3, x4.
        ([(1, (0, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 139),  # not x0, x2, x3, x4.
        ([(1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 140),  # not x1, x2, x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 141),  # x0, not x1, x2, x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 142),  # x0, not x1, x2, x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 143),  # x0, not x1, x3, x4.
        ([(0, (0, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 144),  # x0, not x2, x3, x4.
        ([(0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 145),  # x1, not x2, x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1))], 5, 146),  # not x0, not x1, x2, x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (4, 1))], 5, 147),  # not x0, not x1, x2, x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (3, 1)), (0, (4, 1))], 5, 148),  # not x0, not x1, x3, x4.
        ([(1, (0, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 149),  # not x0, not x2, x3, x4.
        ([(1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 150),  # not x1, not x2, x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 151),  # x0, x1, not x2, x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 152),  # x0, x1, not x2, x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 153),  # x0, x1, not x3, x4.
        ([(0, (0, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 154),  # x0, x2, not x3, x4.
        ([(0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 155),  # x1, x2, not x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 156),  # not x0, x1, not x2, x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 157),  # not x0, x1, not x2, x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 158),  # not x0, x1, not x3, x4.
        ([(1, (0, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 159),  # not x0, x2, not x3, x4.
        ([(1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 160),  # not x1, x2, not x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 161),  # x0, not x1, not x2, x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 162),  # x0, not x1, not x2, x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 163),  # x0, not x1, not x3, x4.
        ([(0, (0, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 164),  # x0, not x2, not x3, x4.
        ([(0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 165),  # x1, not x2, not x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1))], 5, 166),  # not x0, not x1, not x2, x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (4, 1))], 5, 167),  # not x0, not x1, not x2, x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (3, 1)), (0, (4, 1))], 5, 168),  # not x0, not x1, not x3, x4.
        ([(1, (0, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 169),  # not x0, not x2, not x3, x4.
        ([(1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 170),  # not x1, not x2, not x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 171),  # x0, x1, x2, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 172),  # x0, x1, x2, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 173),  # x0, x1, x3, not x4.
        ([(0, (0, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 174),  # x0, x2, x3, not x4.
        ([(0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 175),  # x1, x2, x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 176),  # not x0, x1, x2, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 177),  # not x0, x1, x2, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 178),  # not x0, x1, x3, not x4.
        ([(1, (0, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 179),  # not x0, x2, x3, not x4.
        ([(1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 180),  # not x1, x2, x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 181),  # x0, not x1, x2, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 182),  # x0, not x1, x2, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 183),  # x0, not x1, x3, not x4.
        ([(0, (0, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 184),  # x0, not x2, x3, not x4.
        ([(0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 185),  # x1, not x2, x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1))], 5, 186),  # not x0, not x1, x2, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (4, 1))], 5, 187),  # not x0, not x1, x2, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (3, 1)), (1, (4, 1))], 5, 188),  # not x0, not x1, x3, not x4.
        ([(1, (0, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 189),  # not x0, not x2, x3, not x4.
        ([(1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 190),  # not x1, not x2, x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 191),  # x0, x1, not x2, not x3.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 192),  # x0, x1, not x2, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 193),  # x0, x1, not x3, not x4.
        ([(0, (0, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 194),  # x0, x2, not x3, not x4.
        ([(0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 195),  # x1, x2, not x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 196),  # not x0, x1, not x2, not x3.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 197),  # not x0, x1, not x2, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 198),  # not x0, x1, not x3, not x4.
        ([(1, (0, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 199),  # not x0, x2, not x3, not x4.
        ([(1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 200),  # not x1, x2, not x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 201),  # x0, not x1, not x2, not x3.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 202),  # x0, not x1, not x2, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 203),  # x0, not x1, not x3, not x4.
        ([(0, (0, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 204),  # x0, not x2, not x3, not x4.
        ([(0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 205),  # x1, not x2, not x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1))], 5, 206),  # not x0, not x1, not x2, not x3.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (4, 1))], 5, 207),  # not x0, not x1, not x2, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (3, 1)), (1, (4, 1))], 5, 208),  # not x0, not x1, not x3, not x4.
        ([(1, (0, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 209),  # not x0, not x2, not x3, not x4.
        ([(1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 210),  # not x1, not x2, not x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 211),  # x0, x1, x2, x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 212),  # not x0, x1, x2, x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 213),  # x0, not x1, x2, x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 214),  # not x0, not x1, x2, x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 215),  # x0, x1, not x2, x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 216),  # not x0, x1, not x2, x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 217),  # x0, not x1, not x2, x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (0, (4, 1))], 5, 218),  # not x0, not x1, not x2, x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 219),  # x0, x1, x2, not x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 220),  # not x0, x1, x2, not x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 221),  # x0, not x1, x2, not x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 222),  # not x0, not x1, x2, not x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 223),  # x0, x1, not x2, not x3, x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 224),  # not x0, x1, not x2, not x3, x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 225),  # x0, not x1, not x2, not x3, x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (0, (4, 1))], 5, 226),  # not x0, not x1, not x2, not x3, x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 227),  # x0, x1, x2, x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 228),  # not x0, x1, x2, x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 229),  # x0, not x1, x2, x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 230),  # not x0, not x1, x2, x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 231),  # x0, x1, not x2, x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 232),  # not x0, x1, not x2, x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 233),  # x0, not x1, not x2, x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (0, (3, 1)), (1, (4, 1))], 5, 234),  # not x0, not x1, not x2, x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 235),  # x0, x1, x2, not x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 236),  # not x0, x1, x2, not x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 237),  # x0, not x1, x2, not x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (0, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 238),  # not x0, not x1, x2, not x3, not x4.
        ([(0, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 239),  # x0, x1, not x2, not x3, not x4.
        ([(1, (0, 1)), (0, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 240),  # not x0, x1, not x2, not x3, not x4.
        ([(0, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 241),  # x0, not x1, not x2, not x3, not x4.
        ([(1, (0, 1)), (1, (1, 1)), (1, (2, 1)), (1, (3, 1)), (1, (4, 1))], 5, 242),  # not x0, not x1, not x2, not x3, not x4.
    ]

    def rule_body_list(rule_body):
        for i in range(len(rule_body)):
            rule_body[i] = list(rule_body[i])
            rule_body[i][1] = list(rule_body[i][1])
        return rule_body

    for i in indices:
        idx = get_index(i[0], i[1])
        assert idx == i[2], (
            'get_index({0}, {1}): Expected {2}, actual {3}'.format(
                i[0], i[1], i[2], idx,
            ))
        prog = index_to_prog(idx, i[1])
        assert prog == rule_body_list(i[0]), (
            f'index_to_prog({idx}, {i[1]}): Expected {i[0]}, actual {prog}')
