import glob
import os
import shutil

import numpy as np

from sklearn.model_selection import (
    KFold, LeaveOneGroupOut
)
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.utils import *
from utils.model_utils import *
from utils.train_utils import *

def check_args(n_states: int, hmm_step_type: str):
    if n_states < 3 and hmm_step_type == 'single':
        raise ValueError("HMM with Single Step requires at least 3 states.")
    
    if n_states < 4 and hmm_step_type == 'double':
        raise ValueError("HMM with Double Step requires at least 4 states.")
    
    if n_states < 5 and hmm_step_type == 'triple':
        raise ValueError("HMM with Triple Step requires at least 5 states.")

def train(
    parallel_jobs: int=32,
    train_iters: list=[150],
    leave_one_out: bool=False,
    n_splits: int=15,
    hmm_step_type: str = 'single',
    gmm_mix: int = None,
    gmm_pattern: str = 'middle',
    random_state: int = 4563
):
    print("You have invoked parallel cross validation. Be prepared for dancing progress bars!")
    
    htk_filepaths = glob.glob('data/htk/*.htk')
    phrases = [filepath.split('/')[-1].split(".")[0] + " " + ' '.join(filepath.split('/')[-1].split(".")[1].split("_"))
        for filepath
        in htk_filepaths]
    
    unique_phrases = set(phrases)
    # print(len(unique_phrases), len(phrases))
    group_map = {phrase: i for i, phrase in enumerate(unique_phrases)}
    groups = [group_map[phrase] for phrase in phrases]
    
    if leave_one_out:
        cross_val = LeaveOneGroupOut()
    else:
        cross_val = KFold(n_splits=n_splits)
    
    splits = list(cross_val.split(htk_filepaths, phrases, groups))
    
    mean = 0.0
    variance = 1e-5
    stats = Parallel(n_jobs=parallel_jobs)(
        delayed(crossValFold)(
            np.array(htk_filepaths)[splits[currFold][0]],
            np.array(htk_filepaths)[splits[currFold][1]],
            currFold,
            train_iters,
            mean,
            variance,
            hmm_step_type,
            gmm_mix,
            gmm_pattern,
            random_state
        ) for currFold in range(len(splits))
    )
    
    # all_results['average']['error'] = mean([i[0] for i in stats])
    # all_results['average']['sentence_error'] = mean([i[1] for i in stats])
    # all_results['average']['insertions'] = mean([i[2] for i in stats])
    # all_results['average']['deletions'] = mean([i[3] for i in stats])

