import os
import sys
import glob
import argparse
import shutil
import tqdm
import string
import json

import numpy as np
import pandas as pd

from collections import defaultdict
from io import TextIOWrapper
from numpy.lib.arraysetops import unique
from scipy import stats

from utils.utils import *

########################### Text File Functions ###########################

def generate_text_files(features_dir: str, isFingerspelling: bool, isSingleWord: bool) -> None:
    """Creates all text files needed to train/test HMMs with HTK,
    including wordList, dict, grammar, and all_labels.mlf.
    
    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """

    unique_words = _get_unique_words(features_dir)

    _generate_word_list(unique_words, isFingerspelling)

    _generate_word_dict(unique_words, isFingerspelling)

    _generate_grammar(unique_words, features_dir, isFingerspelling, isSingleWord)

    _generate_mlf_file(isFingerspelling)

    _generate_prototype_file(unique_words)


def _get_unique_words(features_dir: str) -> set:
    """Gets all unique words from a data set.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.

    Returns
    -------
    unique_words : set
        Set of all words found in the training data.
    """

    unique_words = set()
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.split('.')[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        unique_words = unique_words.union(phrase)

    unique_words = sorted(unique_words)
    print("Unique Words: ", unique_words)
    return unique_words


def _generate_word_list(unique_words: list, is_fingerspelling: bool) -> None:
    """Generates wordList file containing all unique words and silences.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    if is_fingerspelling:
        word_list = list(string.ascii_lowercase)
    else:
        word_list = list(unique_words)
    word_list += ['sil0', 'sil1']

    with open('wordList', 'w') as f:
        
        for word in word_list[:-1]:
            f.write('{}\n'.format(word))
        f.write('{}'.format(word_list[-1]))


def _generate_word_dict(unique_words: list, is_fingerspelling: bool) -> None:
    """Generates dict file containing key-value pairs of words. In our
    case, the key and value are both the single, unique word.

    Parameters
    ----------
    unique_words : set
        Set of all words found in the training data.
    """
    
    word_list = list(unique_words)
    word_list += ['sil0', 'sil1']

    with open('dict', 'w') as f:

        f.write('SENT-START [] sil0\n')
        f.write('SENT-END [] sil1\n')
        
        for word in word_list[:-2]:
            if is_fingerspelling:
                f.write('{} {}\n'.format(word, ' '.join(word)))
            else:
                f.write('{} {}\n'.format(word, word))
        f.write('{} {}\n'.format(word_list[-2], word_list[-2]))
        f.write('{} {}\n'.format(word_list[-1], word_list[-1]))
    if is_fingerspelling:
        with open('dict_tri', 'w') as f:
            f.write('SENT-START [] sil0\n')
            f.write('SENT-END [] sil1\n')
            
            for word in word_list[:-2]:
                letters = list(word)
                letters.insert(0, "sil0")
                letters.append("sil1")
                f.write('{} '.format(letters[0]))
                for i in range(len(letters)-2):
                    f.write('{}+{}-{} '.format(letters[i], letters[i+1], letters[i+2]))
                f.write('{}\n'.format(letters[-1]))
            f.write('{} {}\n'.format(word_list[-2], word_list[-2]))
            f.write('{} {}\n'.format(word_list[-1], word_list[-1]))

def _write_grammar_line(
        f: TextIOWrapper, part_of_speech: str, words: list, n='') -> None:
    """Writes a single line to grammar.txt.

    Parameters
    ----------
    f : TextIOWrapper
        Buffered text stream to write to grammar.txt file.

    part_of_speech : str
        Part of speech being written on line.

    words : list
        List of words to be written to line.

    n : str, optional, by default ''
        If a part of speech can be included more than once in the
        grammar, each one should have a distinct count.
    """

    f.write('${}{} = '.format(part_of_speech, n))
    for word in words[:-1]:
        f.write('{} | '.format(word))
    f.write('{};\n'.format(words[-1]))


def _generate_grammar(unique_words: set, features_dir: str, isFingerspelling: bool, isSingleWord: bool) -> None:
    """Creates rule-based grammar depending on the length of the longest
    phrase of the dataset.

    Parameters
    ----------
    features_dir : str
        Unix style pathname pattern pointing to all the features
        extracted from training data.
    """
    if isFingerspelling or isSingleWord:
        with open('grammar.txt', 'w') as f:
            _write_grammar_line(f, 'word', unique_words)
            f.write('\n')
            f.write('(SENT-START $word SENT-END)')
            f.write('\n')
        f.close()
        print("DO")
        return
    
    subjects = set()
    prepositions = set()
    objects = set()
    adjectives = set()
    max_phrase_len = 0
    features_filepaths = glob.glob(os.path.join(features_dir, '**/*.data'), recursive = True)
    features_filepaths.extend(glob.glob(os.path.join(features_dir, '**/*.json'), recursive = True))
    split_index = 1

    for features_filepath in features_filepaths:
        filename = features_filepath.split('/')[-1]
        phrase = filename.split('.')[split_index].split('_')
        phrase = [word.lower() for word in phrase]
        phrase_len = len(phrase)
        max_phrase_len = max(phrase_len, max_phrase_len)

        if phrase_len == 3:

            subject, preposition, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            objects.add(object_)

        elif phrase_len == 4:

            subject, preposition, adjective, object_ = phrase
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective)
            objects.add(object_)

        elif phrase_len == 5:

            adjective_1, subject, preposition, adjective_2, object_ = phrase
            adjectives.add(adjective_1)
            subjects.add(subject)
            prepositions.add(preposition)
            adjectives.add(adjective_2)
            objects.add(object_)

    subjects = list(subjects)
    prepositions = list(prepositions)
    objects = list(objects)
    adjectives = list(adjectives)

    with open('grammar.txt', 'w') as f:
    
        if max_phrase_len == 3:

            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START $subject $preposition $object SENT-END)')
            f.write('\n')
            
        elif max_phrase_len == 4:

           _write_grammar_line(f, 'subject', subjects)
           _write_grammar_line(f, 'preposition', prepositions)
           _write_grammar_line(f, 'adjective', adjectives)
           _write_grammar_line(f, 'object', objects)
           f.write('\n')
           f.write('(SENT-START $subject $preposition [$adjective] $object SENT-END)')
           f.write('\n')

        elif max_phrase_len == 5:

            _write_grammar_line(f, 'adjective', adjectives, 1)
            _write_grammar_line(f, 'subject', subjects)
            _write_grammar_line(f, 'preposition', prepositions)
            _write_grammar_line(f, 'adjective', adjectives, 2)
            _write_grammar_line(f, 'object', objects)
            f.write('\n')
            f.write('(SENT-START [$adjective1] $subject $preposition [$adjective2] $object SENT-END)')
            f.write('\n')

    f.close()


def _generate_mlf_file(isFingerspelling: bool) -> None:
    """Creates all_labels.mlf file that contains every phrase in the 
    dataset.
    """

    htk_filepaths = os.path.join('data', 'htk', '*.htk')
    filenames = glob.glob(htk_filepaths)

    with open('all_labels.mlf', 'w') as f:
        
        f.write('#!MLF!#\n')

        for filename in filenames:

            label = filename.split('/')[-1].replace('htk', 'lab')
            phrase = label.split('.')[1].split('_')

            f.write('"*/{}"\n'.format(label))
            f.write('sil0\n')

            for word in phrase:
                if isFingerspelling:
                    f.write('{}\n'.format('\n'.join(word.lower())))
                else:
                    f.write('{}\n'.format(word.lower()))

            f.write('sil1\n')
            f.write('.\n')

def _generate_prototype_file(unique_words: set):
    words = list(unique_words)
    words.extend(["sil0", "sil1"])
    prototypes = {'8': words}      # '8' is just a dummy choice. it doesn't matter for now.
    
    with open('configs/prototypes.json', 'w', encoding='utf-8') as f:
        json.dump(prototypes, f, ensure_ascii=False, indent=4)

########################### HTK Creation Functions ###########################

def create_htk_files(htk_dir: str = os.path.join('data', 'htk'), ark_dir: str = os.path.join('data', 'ark', '*.ark')) -> None:
    """Converts .ark files to .htk files for use by HTK.
    """
    if os.path.exists(htk_dir):
        shutil.rmtree(htk_dir)

    os.makedirs(htk_dir)

    ark_files = glob.glob(ark_dir)

    for ark_file in tqdm.tqdm(ark_files):
        htk_script_file = os.path.join('/kaldi', 'src/featbin/copy-feats-to-htk')
        kaldi_command = (f'{htk_script_file} '
                         f'--output-dir={htk_dir} '
                         f'--output-ext=htk '
                         f'--sample-period=40000 '
                         f'ark:{ark_file}')
                         # f'>/dev/null 2>&1')

        ##last line silences stdout and stderr
        # print(kaldi_command)
        os.system(kaldi_command)

########################### Ark Creation Functions ###########################

def _create_ark_file(df: pd.DataFrame, ark_filepath: str, title: str) -> None:
    """Creates a single .ark file

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing selected feature.

    ark_filepath : str
        File path at which to save .ark file.

    title : str
        Title containing label needed as header of .ark file.
    """

    with open(ark_filepath, 'w') as out:
        out.write('{} [ '.format(title))

    df.to_csv(ark_filepath, mode='a', header=False, index=False, sep=' ')

    with open(ark_filepath, 'a') as out:
        out.write(']')


def create_ark_files(features_config: dict, users: list, phrase_len: list, verbose: bool, 
                is_select_features: bool, use_optical_flow: bool) -> None:
    """Creates .ark files needed as intermediate step to creating .htk
    files

    Parameters
    ----------
    features_config : dict
        Contains features_dir and features_to_extract

    verbose : bool, optional, by default False
        Whether to print output during process.
    """

    ark_dir = os.path.join('data', 'ark')
    
    if os.path.exists(ark_dir):
        shutil.rmtree(ark_dir)

    os.makedirs(ark_dir)
    if not users:
        features_filepaths = glob.glob(os.path.join(features_config['features_dir'], '**', '*.data'), recursive = True)
        features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '**', '*.json'), recursive = True))
    else:
        features_filepaths = []
        print(users)
        for user in users:
            print(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.json'))
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.data'), recursive = True))
            features_filepaths.extend(glob.glob(os.path.join(features_config['features_dir'], '*{}_*'.format(user), '**', '*.json'), recursive = True))
    
    features_filepaths = list(filter(lambda x: len(x.split('.')[1].split('_')) in phrase_len, features_filepaths))
    
    print(features_config['features_dir'])
    if is_select_features:
        print("Generating ark/htk using select_features data model")
    else:
        print("Generating ark/htk using interpolate_features data model")
    
    features_rows = defaultdict(int)
    for features_filepath in tqdm.tqdm(features_filepaths):

        if verbose:
            print(features_filepath)

        features_filename = features_filepath.split('/')[-1]
        features_extension = features_filename.split('.')[-1]
        features_df = None

        ark_filename = features_filename.replace(features_extension, 'ark')
        ark_filepath = os.path.join(ark_dir, ark_filename)
        title = ark_filename.replace('.ark', "")
        
        if is_select_features:
            features_df = select_features(features_filepath, features_config['selected_features'], center_on_nose = True, scale = 100, square = True, 
                                    drop_na = True, do_interpolate = True, use_optical_flow=use_optical_flow)
        else:
            features_df = interpolate_feature_data(features_filepath, features_config['selected_features'], center_on_face = False, is_2d = True, scale = 10, drop_na = True)
        
        if features_df is not None:
            num_rows = features_df.shape[0]
            if num_rows > 0:
                _create_ark_file(features_df, ark_filepath, title)
            features_rows[num_rows] += 1
    print("Features Num Rows Distribution: ", features_rows)
        
