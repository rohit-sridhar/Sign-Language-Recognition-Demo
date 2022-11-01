import os
import glob

from tqdm import tqdm

from utils.utils import *
from utils.model_utils import *

import numpy as np

#################################################### Train Functions ####################################################

def crossValFold(
    train_data: list,
    test_data: list,
    fold: int,
    train_iters: list=[150],
    mean: float=0.0,
    variance: float=1e-5,
    hmm_step_type: str = 'single',
    gmm_mix: int = None,
    gmm_pattern: str = 'middle',
    random_state: int = 4563,
    run_train = True,
):
    train_data = np.array(train_data)
    np.random.seed(random_state)
    np.random.shuffle(train_data)

    info_string = (
        f"Current split = {str(fold)}. "
        f"Current Test data Size = {len(test_data)}. "
        f"Current Train data Size = {len(train_data)}."
    )
    print(info_string)
    ogDataFolder = "data"
    currDataFolder = os.path.join("data", str(fold))
    trainFiles = [i.split("/")[-1].replace(".htk", "") for i in train_data]
    testFiles = [i.split("/")[-1].replace(".htk", "") for i in test_data]
    allFiles = trainFiles + testFiles

    copyFiles(allFiles, os.path.join(currDataFolder, "ark"), os.path.join(ogDataFolder, "ark"), ".ark")
    copyFiles(allFiles, os.path.join(currDataFolder, "htk"), os.path.join(ogDataFolder, "htk"), ".htk")
    create_data_lists([os.path.join(currDataFolder, "htk", i+".htk") for i in trainFiles], [
                    os.path.join(currDataFolder, "htk", i+".htk") for i in testFiles], fold)
    
    if run_train:
        _train(
            train_iters,
            mean,
            variance,
            fold=os.path.join(str(fold), ""),
            hmm_step_type=hmm_step_type,
            gmm_mix=gmm_mix,
            gmm_pattern=gmm_pattern
        )
    # test(args.start, args.end, args.method, args.hmm_insertion_penalty, fold=os.path.join(str(fold), ""))

    # hresults_file = f'hresults/{os.path.join(str(fold), "")}res_hmm{args.train_iters[-1]-1}.txt'
    # results = get_results(hresults_file)

    # print(f'Current Word Error: {results["error"]}')
    # print(f'Current Sentence Error: {results["sentence_error"]}')
    # print(f'Current Insertion Error: {results["insertions"]}')
    # print(f'Current Deletions Error: {results["deletions"]}')

    # # test(-1, -1, "alignment", args.hmm_insertion_penalty, beam_threshold=args.beam_threshold, fold=os.path.join(str(fold), ""))

    # return [results['error'], results['sentence_error'], results['insertions'], results['deletions']]

def _train(
    train_iters: list,
    mean: float,
    variance: float,
    fold: str = "",
    hmm_step_type: str = 'single',
    gmm_mix: int = None,
    gmm_pattern: str = 'middle'
) -> None:
    """Trains the HMM using HTK. Calls HCompV, HRest, HERest, HHEd, and
    HParse. Configuration files for prototypes and increasing mixtures
    are found in configs/. 

    Parameters
    ----------
    train_args : Namespace
        Argument group defined in train_cli() and split from main
        parser.
    """

    if os.path.exists(f'models/{fold}'):
        shutil.rmtree(f'models/{fold}')

    if os.path.exists(f'logs/{fold}'):
        if os.path.exists(f'logs/{fold}train.log'):
            os.remove(f'logs/{fold}train.log')

    os.makedirs(f'models/{fold}')

    if not os.path.exists(f'logs/{fold}'):
        os.makedirs(f'logs/{fold}')

    #n_models = train_iters[-1] + len(train_iters) - 1
    for i in range(train_iters[-1] + 1):
        hmm_dir = os.path.join('models', f'{fold}hmm{i}')
        if not os.path.exists(hmm_dir):
            os.makedirs(hmm_dir)

    features_config = load_json('configs/features.json')
    n_features = len(features_config['selected_features'])

    print("-------------- Training HMM --------------")

    prototypes_config = load_json('configs/prototypes.json')
    for n_states in prototypes_config:
        n_states_int = int(n_states)
        prototype_filepath = f'models/{fold}prototype'
        generate_prototype(
            n_states_int, n_features, prototype_filepath, mean,
            variance, hmm_step_type=hmm_step_type, gmm_mix=gmm_mix,
            gmm_pattern=gmm_pattern
        )

        print('Running HCompV...')
        HCompV_command = (f'HCompV -A -T 2 -C configs/hcompv.conf -v 2.0 -f 0.01 '
                          f'-m -S lists/{fold}train.data -M models/{fold}hmm0 '
                          f'{prototype_filepath} >> logs/{fold}train.log')
        os.system(HCompV_command)
        print('HCompV Complete')

        initialize_models(f'models/{fold}hmm0/prototype', prototypes_config[n_states], f'models/{fold}hmm0')
        #initialize_models('models/prototype', 'wordList', 'models/hmm0')

    hmm0_files = set(glob.glob(f'models/{fold}hmm0/*')) - {f'models/{fold}hmm0/vFloors'}
    for hmm0_file in tqdm(hmm0_files):

        # print(f'Running HRest for {hmm0_file}...')
        HRest_command = (f'HRest -A -i 60 -C configs/hrest.conf -v 0.1 -I '
                         f'all_labels.mlf -M models/{fold}hmm1 -S lists/{fold}train.data '
                         f'{hmm0_file} >> logs/{fold}train.log')
        os.system(HRest_command)
    print('HRest Complete')

    print('Running HERest Iteration: 1...')
    HERest_command = (f'HERest -A -d models/{fold}hmm1 -c 500.0 -v 0.0005 -I '
                    f'all_labels.mlf -M models/{fold}hmm2 -S lists/{fold}train.data -T '
                    f'1 wordList >> logs/{fold}train.log')
    os.system(HERest_command)
    
    start = 2
    for i, n_iters in enumerate(train_iters):

        for iter_ in tqdm(range(start, n_iters)):

            # print(f'Running HERest Iteration: {iter_}...')
            HERest_command = (f'HERest -A -c 500.0 -v 0.0005 -A -H '
                        f'models/{fold}hmm{iter_}/newMacros -I all_labels.mlf -M '
                        f'models/{fold}hmm{iter_+1} -S lists/{fold}train.data -T 1 wordList '
                        f'>> logs/{fold}train.log')
            os.system(HERest_command)
        print('HERest Complete')

        if n_iters != train_iters[-1]:
            print(f'Running HHed Iteration: {n_iters}...')
            HHed_command = (f'HHEd -A -H models/{fold}hmm{n_iters-1}/newMacros -M '
                        f'models/{fold}hmm{n_iters} configs/hhed{i}.conf '
                        f'wordList')
            os.system(HHed_command)
            print('HHed Complete')
            start = n_iters

    cmd = 'HParse -A -T 1 grammar.txt wordNet.txt'
    os.system(cmd)

