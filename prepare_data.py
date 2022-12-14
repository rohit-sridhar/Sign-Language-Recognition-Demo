"""Prepares training data. Creates .ark files, .htk files, wordList,
dict, grammar, and all_labels.mlf.

Methods
-------
prepare_data
"""

from utils.utils import *
from utils.data_creation_utils import *

def prepare_data(
    features_config: dict,
    users: list=None,
    phrase_len:list=[3,4,5],
    prediction_len:list=[3,4,5],
    isFingerspelling:bool=False,
    isSingleWord:bool=False
) -> None:

    """Prepares training data. Creates .ark files, .htk files, wordList,
    dict, grammar, and all_labels.mlf.

    Parameters
    ----------
    features_config : dict
        A dictionary defining which features to use when creating the 
        data files.
    """
    create_ark_files(features_config, users, [1], verbose=False, is_select_features=True, use_optical_flow=False)
    print('.ark files created')

    print('Creating .htk files')
    create_htk_files()
    print('.htk files created')

    print('Creating .txt files')
    generate_text_files(features_config["features_dir"], isFingerspelling, isSingleWord)
    print('.txt files created')
    
    # print("Data already generated, skipping data generation")
