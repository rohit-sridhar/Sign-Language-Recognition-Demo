import os
import random

import numpy as np

################################## HMM Gen Helpers ##################################

# Helper for out prob computation. Assumes self transition is passed in and uniformity among split.
def get_out_prob(self_transition_prob: float, n_split: int):
    return (1 - self_transition_prob) / n_split

# Generates last Single/Double/Triple step row
def generate_last_single_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 2) + [str(transition_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_last_double_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 3) + [str(transition_prob), str(out_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_last_triple_step(f, n_states: int, transition_prob: float, out_prob: float):
    row = ['0.0']*(n_states - 4) + [str(transition_prob), str(out_prob), str(out_prob), str(out_prob)]
    f.write(' '.join(row) + '\n')

def generate_uniform_row(f, n_states: int, i: int = 1):
    unif_prob = round(1.0 / (n_states - i), 2)
    last_prob = round(1.0 - (unif_prob * (n_states - i - 1)), 2)

    row = ['0.0']*i + [str(unif_prob)]*(n_states - i - 1) + [str(last_prob)]
    f.write(' '.join(row) + '\n')

def generate_gmm_state_space(
    f, n_states: int, n_features: int, n_mixes: int, variance: float, i: int
):
    f.write('<State> {} <NumMixes> {}\n'.format(i, n_mixes))
    for j in range(1, n_mixes + 1):
        f.write('<Mixture> {} {}\n'.format(j, 1.0 / n_mixes))
        f.write('<Mean> {}\n'.format(n_features))
        f.write(' '.join([str(random.random()) for _ in range(n_features)]) + '\n')
        f.write('<Variance> {}\n'.format(n_features))
        f.write(' '.join([str(variance)]*n_features) + '\n')

def generate_state_space(
    f, n_states: int, n_features: int, mean: float,
    variance: float, i: int
):
    f.write('<State> {}\n'.format(i))
    f.write('<Mean> {}\n'.format(n_features))
    f.write(' '.join([str(mean)]*n_features) + '\n')
    f.write('<Variance> {}\n'.format(n_features))
    f.write(' '.join([str(variance)]*n_features) + '\n')

#################################################### HMM GEN FUNCTIONS ####################################################

def generate_single_step_hmm(f, n_states: int, transition_prob: float = 0.6, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 1)
    
    for i in range(start, n_states-2):
        row = ['0.0']*i + [str(transition_prob), str(out_prob)] + ['0.0']*(n_states - i - 2)
        f.write(' '.join(row) + '\n')
    
    generate_last_single_step(f, n_states, transition_prob, out_prob)

def generate_double_step_hmm(f, n_states: int, transition_prob: float = 0.4, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 2)
    for i in range(start, n_states-3):
        row = ['0.0']*i + [str(transition_prob), str(out_prob), str(out_prob)] + ['0.0']*(n_states - i - 3)
        f.write(' '.join(row) + '\n')

    generate_last_double_step(f, n_states, transition_prob, out_prob)
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))
    
def generate_triple_step_hmm(f, n_states: int, transition_prob: float = 0.28, start: int = 1) -> None:
    out_prob = get_out_prob(transition_prob, 3)
    for i in range(start, n_states-4):
        row = ['0.0']*i + [str(transition_prob), str(out_prob), str(out_prob), str(out_prob)] + ['0.0']*(n_states - i - 4)
        f.write(' '.join(row) + '\n')

    generate_last_triple_step(f, n_states, transition_prob, out_prob)
    generate_last_double_step(f, n_states, 0.4, get_out_prob(0.4, 2))
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))

def generate_start_stack_hmm(f, n_states: int, transition_prob: float=0.4) -> None:
    out_prob = get_out_prob(transition_prob, 2)
    for i in range(1, n_states-3):
        row = ['0.0']*i + [str(transition_prob), str(out_prob)] + ['0.0']*(n_states - i - 3) + [str(out_prob)]
        f.write(' '.join(row) + '\n')

    generate_last_double_step(f, n_states, transition_prob, out_prob)
    generate_last_single_step(f, n_states, 0.6, get_out_prob(0.6, 1))

def generate_end_stack_hmm(f, n_states: int, transition_prob: float=0.6) -> None:
    generate_uniform_row(f, n_states)
    generate_single_step_hmm(f, n_states=n_states, transition_prob=transition_prob, start = 2)


def generate_prototype(n_states: int, n_features: int, output_filepath: str, 
                       mean: float = 0.0, variance: float = 1.0, 
                       transition_prob: float = 0.6, hmm_step_type: str = 'single',
                       gmm_mix: int = None, gmm_pattern: str = 'middle') -> None:
    """Generates prototype files used to initalize models.

    Parameters
    ----------
    n_states : int
        Number of states each model has.

    n_features : int
        Number of features being used to train each model.

    output_filepath : str
        File path at which to save prototype.

    mean : float, optional, by default 0.0
        Initial value to use as mean of all features.

    variance : float, optional, by default 1.0
        Initial value to use as variance of all features.

    transition_prob : float, optional, by default 0.6
        Initial probability of transition from one state to the next.
    """

    with open(output_filepath, 'w') as f:

        f.write('~o\n')
        f.write('<VecSize> {} <USER>\n'.format(n_features))
        f.write('~h "prototype"\n')
        f.write('<BeginHMM>\n')
        f.write('<NumStates> {}\n'.format(n_states))

        start = 2
        for i in range(start, n_states):
            if gmm_mix is not None:
                if gmm_pattern == 'middle' and i > start + 1 and i < n_states - 2:
                    generate_gmm_state_space(f, n_states, n_features, gmm_mix, variance, i)
                elif gmm_pattern == 'all':
                    generate_gmm_state_space(f, n_states, n_features, gmm_mix, variance, i)
                else:
                    generate_state_space(f, n_states, n_features, mean, variance, i)
            else:
                generate_state_space(f, n_states, n_features, mean, variance, i)

        f.write('<TransP> {}\n'.format(n_states))
        row = ['0.0'] + ['1.0'] + ['0.0']*(n_states - 2)
        f.write(' '.join(row) + '\n')
        
        if hmm_step_type == 'single':
            generate_single_step_hmm(f, n_states)
        elif hmm_step_type == 'double':
            generate_double_step_hmm(f, n_states)
        elif hmm_step_type == 'triple':
            generate_triple_step_hmm(f, n_states)
        elif hmm_step_type == 'start_stack':
            generate_start_stack_hmm(f, n_states)
        elif hmm_step_type == 'end_stack':
            generate_end_stack_hmm(f, n_states)

        f.write(' '.join(['0.0']*n_states) + '\n')
        f.write('<EndHMM>\n')

def initialize_models(
        prototype_filepath: str, words: list, hmm_dir: str) -> None:
    """Initializes models for words in training set from a given
    prototype. Should be called after generate_prototype and in 
    conjuncion with configs/prototypes.json.

    Parameters
    ----------
    prototype_filepath : str
        File path prototype to be used to initialize model.

    words : list
        All words to be initalized with given prototype.

    hmm_dir : str
        Directory at which to save newly created models.
    """

    with open(prototype_filepath, 'r') as f:
        prototype = f.read().strip('\r\n')

    for word in words:

        word = word.strip('\r\n')
        hmm_word_filepath = os.path.join(hmm_dir, word)

        with open(hmm_word_filepath, 'w') as f:
            f.write(prototype.replace('prototype', word))
