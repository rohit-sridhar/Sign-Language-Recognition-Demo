import json
import os
import shutil

import numpy as np
import pandas as pd


def copyFiles(fileNames: list, newFolder: str, originalFolder: str, ext: str):
    if os.path.exists(newFolder):
        shutil.rmtree(newFolder)
    os.makedirs(newFolder)

    for currFile in fileNames:
        shutil.copyfile(os.path.join(originalFolder, currFile+ext), os.path.join(newFolder, currFile+ext))


def create_data_lists(train_data: list, test_data: list, fold: int = None) -> None:
	"""Creates train.data, test.data, and all.data, which contain the
	phrases in the dataset.

	Parameters
	----------
	train_data : list
		All phrases to be included as training data.

	test_data : list
		All phrases to be included as testing data.
	"""
	masterFolder = 'lists' if fold is None else f'lists/{str(fold)}'
	if not os.path.exists(masterFolder):
		os.makedirs(masterFolder)

	all_data_filepath = os.path.join(masterFolder, 'all.data')
	train_data_filepath = os.path.join(masterFolder, 'train.data')
	test_data_filepath = os.path.join(masterFolder, 'test.data')

	with open(all_data_filepath, 'w') as all_data_list, \
		 open(train_data_filepath, 'w') as train_data_list, \
		 open(test_data_filepath, 'w') as test_data_list:

			for phrase in train_data:
				all_data_list.write('{}\n'.format(phrase))
				train_data_list.write('{}\n'.format(phrase))

			for phrase in test_data:
				all_data_list.write('{}\n'.format(phrase))
				test_data_list.write('{}\n'.format(phrase))

def get_results(results_file: str) -> dict:
    """Parses hresults file and returns them as dict.

    Parameters
    ----------
    results_file : str
        File path of hresults file to be parsed. Generally the last in
        a group of trained models.

    Returns
    -------
    results_dict : dict
        Results containing substitutions, deletions, insertions, error,
        and sentence error.
    """

    results_dict = {}
    confusion_string = ""

    with open(results_file, 'r') as lf:
        start_file = False
        for line in lf:
            l = line.rstrip()
            if l == "    ,-------------------------------------------------------------.":
                start_file = True
            if start_file:
                confusion_string += l + "\n"
                if "| Sum/Avg |" in l:
                    vals = l.split("|")[3].strip().split()
                    results_dict['substitutions'] = float(vals[1])
                    results_dict['deletions'] = float(vals[2])
                    results_dict['insertions'] = float(vals[3])
                    results_dict['error'] = float(vals[4])
                    results_dict['sentence_error'] = float(vals[5])
                    
        results_dict['confusion_matrix'] = confusion_string

    return results_dict

def get_num_lines(filename: str) -> int:
    with open(filename, 'r') as f:
        lines = f.readlines()
    return len(lines)

def clean_and_make_path(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def load_json(json_file: str) -> dict:
    """Load JSON file TODO: Remove and use src.utils.load_json.

    Parameters
    ----------
    json_file : str
        File path of JSON to be loaded.

    Returns
    -------
    data : dict
        Data loaded from JSON.
    """
    
    with open(json_file, 'r') as data_file:
        data = json.loads(data_file.read())
        
    return data


def _calc_delta(col: str) -> np.ndarray:
    """Calculates delta between consecutives rows of a given column.

    Parameters
    ----------
    col : str
        Column for which to calculate delta.

    Returns
    -------
    np.ndarray
        Delta of rows.
    """
    
    return np.concatenate([[0], col[1:].values - col[:-1].values])


def _add_delta_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Calculate delta for a column and add it as a new column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing features.

    col : str
        Column for which to calculate delta.

    Returns
    -------
    return_val : pd.DataFrame
        DataFrame containing features with new delta column.
    """
    
    df['delta_{}'.format(col)] = _calc_delta(df[col])
    
    return df.copy()

def landmark_box_dist(landmark: list, hand: list) -> float:

    curr_landmark = np.reshape(landmark, (3,21))
    total_dist = 0
    for point in curr_landmark:
        hand_point = [[hand[0], hand[1]]]
        landmark_point = [[point[0], point[1]]]
        total_dist += cdist(hand_point, landmark_point)[0]
    
    return total_dist/(21)


def select_features(input_filepath: str, features_to_extract: list,
                    interpolation_method: str = 'linear', order: int = 3,
                    center_on_nose: bool = False, center_on_pelvis: bool = False, is_2d: bool = True,
                    scale: int = 10, drop_na: bool = True, do_interpolate: bool = False, use_optical_flow = False,
                    square: bool = False) -> pd.DataFrame:
    """Processes raw features extracted from MediaPipe/Kinect, and
    selects the specified features for use during training of HMMs.

    Parameters
    ----------
    input_filepath : str
        File path of raw feature data to be processed and used for
        selection.

    features_to_extract : list
        Names of columns to be selected after processing features.

    interpolation_method : str, optional, by default 'spline'
        Interpolation method used to fill missing values.

    order : int, optional, by default 3
        Hyperparameter needed for certain interpolation methods.

    center_on_nose : bool, optional, by default False
        Whether to center the features on the main face.

    is_2d : bool, optional, by default True
        Whether data is 2-dimensional.

    scale : int, optional, by default 10
        Raw features are scaled from 0-1, which can cause issues with
        very small means and variances in HTK. Used to scale up
        features.

    Returns
    -------
    df : pd.DataFrame
        Selected features
    """
    data = load_json(input_filepath)
    # print("Input Filepath: ", input_filepath)
    # print("Data Keys: ", data['1'].keys())

    if not data:
        return None
    data = {int(key): value for key, value in data.items()}

    n_frames = len(data)
    hands = np.zeros((2, n_frames, 18))
    landmarks = np.zeros((2, n_frames, 63))
    noses = np.zeros((1, n_frames, 3))
    optical_flow = np.zeros((2, n_frames, 32))
    # print("Features to Extract: ", features_to_extract)

    for frame in sorted(data.keys()):

        if use_optical_flow and data[frame]['optical_flow'] is not None:

            x_val = data[frame]['optical_flow']["0"]
            y_val = data[frame]['optical_flow']["1"]

            optical_flow[0, frame, :] = np.array(x_val)/10000000.0
            optical_flow[1, frame, :] = np.array(y_val)/10000000.0
        
        # print('File: ', input_filepath)
        # if data[frame]["landmarks"]:
        #     print("Landmarks Available")
        # else:
        #     print("Landmarks Unavailable")

        # for hand in data[frame]["landmarks"]:
        #     print("Hand: ", hand)
        #     print('Data Frame Info: ', len(data[frame]['landmarks'][hand]))
        #     print()

        if data[frame]['pose']:

            right_hand = np.array([data[frame]['pose'][str(i)] for i in range(11,22,2)]).reshape((-1,))
            left_hand = np.array([data[frame]['pose'][str(i)] for i in range(12,23,2)]).reshape((-1,))
            curr_nose = np.array(data[frame]['pose']["0"]).reshape((-1,))
            hands[0, frame, :] = right_hand
            hands[1, frame, :] = left_hand
            noses[0, frame, :] = curr_nose
        
        if data[frame]['landmarks']:
            if data[frame]['pose'] is None:
                raise Exception('Red Alert: Our assumption that landmarks are only provided when we have boxes is incorrect')
            else:
                visible_landmarks = [data[frame]['landmarks']["0"], data[frame]['landmarks']["1"]]
                if visible_landmarks[0]:
                    left_landmark = np.array([visible_landmarks[0][str(i)] for i in range(21)]).reshape((-1,))
                    landmarks[1, frame, :] = left_landmark
                if visible_landmarks[1]:
                    right_landmark = np.array([visible_landmarks[1][str(i)] for i in range(21)]).reshape((-1,))
                    landmarks[0, frame, :] = right_landmark
                    
    select_hands = np.any(['hand' 
                           in feature 
                           for feature 
                           in features_to_extract])
    select_landmarks = np.any(['landmark' 
                               in feature 
                               for feature 
                               in features_to_extract])
    select_nose = np.any(['nose' 
                                in feature
                                for feature
                                in features_to_extract]) 

    select_optical_flow = np.any(['optical_flow'
                                in feature
                                for feature
                                in features_to_extract])

    if select_hands and not np.any(hands):
        return None

    if select_landmarks and not np.any(landmarks):
        return None

    if select_nose and not np.any(noses):
        return None
    
    if use_optical_flow and select_optical_flow and not np.any(optical_flow):
        return None

    hands_ = ['right', 'left']
    landmarks_ = ['shoulder', 'elbow', 'wrist', 'pinky', 'index', 'thumb']
    coordinates = ['x', 'y', 'z']
    hand_cols = [f'{hand}_{landmark}_{coordinate}' 
                for hand 
                in hands_
                for landmark
                in landmarks_
                for coordinate 
                in coordinates]

    hands_ = ['right', 'left']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y', 'z']
    landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in landmarks_ 
                    for coordinate 
                    in coordinates]

    nose_ = ['nose']
    coordinates = ['x', 'y', 'z']
    nose_cols = ['{}_{}'.format(nose, coordinate)
                for nose
                in nose_
                for coordinate
                in coordinates]
    
    coordinates = ['optical_flow_x', 'optical_flow_y']
    optical_flow_cols = ['{}_{}'.format(coordinate, i)
                for i
                in range(32)
                for coordinate
                in coordinates]

    # ipdb.set_trace()
    cols = hand_cols + landmark_cols + nose_cols + optical_flow_cols
    hands = np.concatenate([hands[0], hands[1]], axis=1)
    left_wrist = np.tile(landmarks[0,:,0:3], 21)
    right_wrist = np.tile(landmarks[1,:,0:3], 21)
    landmarks = np.concatenate([landmarks[0], landmarks[1]], axis=1)
    centered_landmarks = landmarks - np.concatenate([left_wrist, right_wrist], axis=1)
    optical_flow = np.concatenate([optical_flow[0], optical_flow[1]], axis=1)
    noses = np.reshape(noses, (-1, 3))
    all_features = np.concatenate([hands, landmarks, noses, optical_flow], axis=1)
    df = pd.DataFrame(all_features, columns=cols)
    df_lm = df.loc[:, df.columns.isin(features_to_extract)]
    # print("DF Shape (1): ", df.shape)
    df = df.replace(0, np.nan)
    df_lm = df_lm.replace(0, np.nan)
    # print("DF Shape (2): ", df.shape)

    if select_hands and do_interpolate:

        try:
            df[hand_cols] = df[hand_cols].interpolate(interpolation_method, order=order)
        except Exception as e:
            print(df[hand_cols])
            print(input_filepath)
            return None

    if select_landmarks and do_interpolate:

        try:
            df[landmark_cols] = df[landmark_cols].interpolate(interpolation_method, order=order)
        except Exception as e:
            print(df[landmark_cols])
            print(e)
            print(input_filepath)
            return None

    if center_on_nose:
        
        x_cols = [column for column in df.columns if '_x' in column]
        y_cols = [column for column in df.columns if '_y' in column]
        z_cols = [column for column in df.columns if '_z' in column]

        # SHAPES:
        # df[x_cols]: (159, 87)
        # noses: (159, 3)
        # noses[0]: (,3)
        # noses[:,0]: (159,)
        
        # this is what we had before
        # df[x_cols] -= np.mean(noses, axis=1)[0]
        # df[y_cols] -= np.mean(noses, axis=1)[1]
        # df[z_cols] -= np.mean(noses, axis=1)[2]

        df[x_cols] = (df[x_cols].transpose() - noses[:,0]).transpose()
        df[y_cols] = (df[y_cols].transpose() - noses[:,1]).transpose()
        df[z_cols] = (df[z_cols].transpose() - noses[:,2]).transpose()

    # print("DF Shape (3): ", df.shape)
    df['horizontal_hand_dist'] = df['right_index_x'] - df['left_index_x']
    df['vertical_hand_dist'] = df['right_index_y'] - df['left_index_y']

    hands_ = ['right', 'left']
    centered_landmarks_ = ['centered_landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y', 'z']
    centered_landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in centered_landmarks_ 
                    for coordinate 
                    in coordinates]
    wrist = ['{}_{}_{}'.format(hand, wrist_landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for wrist_landmark
                    in ['landmark_0' for i in range(21)]
                    for coordinate 
                    in coordinates]
    for old_lm, centered_lm, wrist_coord in zip(landmark_cols, centered_landmark_cols, wrist):
        df[centered_lm] = df[old_lm] - df[wrist_coord]

    for col in df.columns:

        df = _add_delta_col(df, col)

    df = df.loc[:, df.columns.isin(features_to_extract)]
    # ipdb.set_trace()
    # print("DF Shape (4): ", df.shape)
    if drop_na:
        df = df.dropna(axis=0)
    # print("DF Shape (5): ", df.shape)
    df = df * scale
    if square:
        df = pd.DataFrame(df.values*df.abs().values, columns=df.columns, index=df.index)
    df = df.round(6)
    
    # print("DF Shape (Last): ", df.shape)
    # print()
    return df

def interpolate_feature_data(features_filepath, features, center_on_face: bool = False, is_2d: bool = True, scale: int = 10, drop_na: bool = True):

    """Processes raw features extracted from MediaPipe, assigns an ID to track the features, and
    then interpolates using averages for the specified features for visualization.

    Parameters
    ----------
    features_filepath : str
        File path of raw mediapipe feature data to be processed

    features : list of str
        Names of columns to be selected after processing features.

    Returns
    -------
    df : pd.DataFrame
        Selected features from mediapipe with interpolation

    TODO:
    -----
    Split up the assign_hand_feature_id() into more functions as a lot of the code is repeated
    Merge the different types of interpolations (to a degree)
    Attempt to pythonify the creation of the dataframe 
    Add more parameters that can impact which features are selected from dataframe (look at feature_selection.py), and possibly allow usage for 3D data.

    """

    ordered_hands = assign_hand_feature_id(features_filepath, False, 'boxes') # False means 3 boxes will be used in tracking

    ordered_landmarks = assign_hand_feature_id(features_filepath, False, 'landmarks')
    
    ordered_faces = assign_face_feature_id(features_filepath, 'faces')
    
    #print("Finished Assignment")

    ordered_hands = interpolate_hand(ordered_hands)

    ordered_landmarks = interpolate_landmarks(ordered_landmarks)

    ordered_faces = interpolate_faces(ordered_faces)

    #print("Finished Interpolation")

    num_frames = len(ordered_hands)

    # Create Dataframe from data
    hand_0 = []
    hand_1 = []
    for i in range(num_frames):
        frame_hand_data_0 = ordered_hands[i][0]
        frame_hand_data_1 = ordered_hands[i][1]
        hand_0.append(frame_hand_data_0)
        hand_1.append(frame_hand_data_1)

    hand_0 = np.array(hand_0)
    hand_1 = np.array(hand_1)

    landmarks_0 = []
    landmarks_1 = []
    for i in range(num_frames):
        frame_landmark_data_0 = ordered_landmarks[i][0]
        frame_landmark_data_1 = ordered_landmarks[i][1]
        data_0 = []
        data_1 = []
        for j in range(0, 21):
            data_0.append(frame_landmark_data_0[j][0])
            data_0.append(frame_landmark_data_0[j][1])
            data_1.append(frame_landmark_data_1[j][0])
            data_1.append(frame_landmark_data_1[j][1])
        landmarks_0.append(data_0)
        landmarks_1.append(data_1)

    landmarks_0 = np.array(landmarks_0)
    landmarks_1 = np.array(landmarks_1)

    face_0 = []
    for i in range(num_frames):
        frame_face_data_0 = ordered_faces[i]
        data_0 = []
        for j in range(0, 6):
            data_0.append(frame_face_data_0[j][0])
            data_0.append(frame_face_data_0[j][1])
        face_0.append(data_0)

    face_0 = np.array(face_0)

    hands_ = ['left_hand', 'right_hand']
    coordinates = ['x', 'y', 'w', 'h']
    hand_cols = [f'{hand}_{coordinate}' 
                for hand 
                in hands_ 
                for coordinate 
                in coordinates]

    hands_ = ['left', 'right']
    landmarks_ = ['landmark_{}'.format(i) for i in range(21)]
    coordinates = ['x', 'y']
    landmark_cols = ['{}_{}_{}'.format(hand, landmark, coordinate) 
                    for hand 
                    in hands_ 
                    for landmark 
                    in landmarks_ 
                    for coordinate 
                    in coordinates]


    faces_ = ['face_{}'.format(i) for i in range(6)]
    coordinates = ['x', 'y']
    face_cols = ['{}_{}'.format(face, coordinate)
                for face
                in faces_
                for coordinate
                in coordinates]

    cols = hand_cols + landmark_cols + face_cols

    all_features = np.concatenate([hand_0, hand_1, landmarks_0, landmarks_1, face_0], axis=1)

    df = pd.DataFrame(all_features, columns=cols)

    df = df.replace(0, np.nan)

    if is_2d:
        
        z_landmark_cols = [column for column in landmark_cols if 'z' in column]
        df = df.drop(z_landmark_cols, axis=1)

    if center_on_face:
        
        x_cols = [column for column in df.columns if 'x' in column]
        y_cols = [column for column in df.columns if 'y' in column]

        x_faces = [column for column in df.columns if 'x' in column and 'face' in column]
        y_faces = [column for column in df.columns if 'y' in column and 'face' in column]

        for frame in range(num_frames):
            avg_x = df.loc[frame, x_faces].mean()
            avg_y = df.loc[frame, y_faces].mean() 

            df.loc[frame, x_cols] -= avg_x
            df.loc[frame, y_cols] -= avg_y

    df['horizontal_hand_dist'] = df['right_hand_x'] - df['left_hand_x']
    df['vertical_hand_dist'] = df['right_hand_y'] - df['left_hand_y']

    for col in df.columns:

        df = _add_delta_col(df, col)

    df = df.loc[:, df.columns.isin(features)]
    if drop_na: df = df.dropna(axis=0)
    df = df * scale
    df = df.round(6)    

 

    return df

def iou_threshold(threshold, no_data_streak):
    return threshold + no_data_streak * 0.01

def dist_boxes(point1, point2):
    xDist = (point2[0] - point1[0]) * (point2[0] - point1[0]) 
    yDist = (point2[1] - point1[1]) * (point2[1] - point1[1])
    return xDist + yDist

def dist_landmarks(points1, points2): # for landmarks and eventually faces.. similar to currently interpolation method, except I do not need to call center_bbox_point, and dist is calculated like this. Make sure maps/array match up correctly
    avg = 0
    size = len(points1)
    for i in range(size):
        x1, y1 = points1[str(i)][0:2]
        x2, y2 = points2[i][0:2]
        dist = dist_boxes([x1, y1], [x2, y2])
        avg += dist
    avg /= float(size)
    return avg

def calculate_distance(data1, data2, feature_type):
    if feature_type == 'landmarks' or feature_type == 'faces':
        return dist_landmarks(data1, data2)
    elif feature_type == 'boxes':
        return dist_boxes(data1, data2)
    return None

def center_bbox_point(bbox):
    xPos = bbox[0] + bbox[2]/2
    yPos = bbox[1] + bbox[3]/2
    return [xPos, yPos]

def calculate_center(data, feature_type):
    if feature_type == 'landmarks' or feature_type == 'faces':
        return data
    elif feature_type == 'boxes':
        return center_bbox_point(data)
    return None

def set_interpolated_data(interpolated_data, frame, id_assign, curr_data, feature_type):
    if feature_type == 'landmarks':
        for elem in range(0, 21):
            interpolated_data[frame][id_assign][elem] = curr_data['{}'.format(elem)]
    elif feature_type == 'boxes':
        interpolated_data[frame][id_assign] = curr_data

def interpolate_faces(ordered):
    # Only supports faces

    num_points = 6

    prev_pnt = 0 # first location with data
    next_pnt = 0 # first next location with data
    frames = len(ordered)

    while next_pnt < frames and sum(ordered[next_pnt][0]) == 0:
        next_pnt += 1
        prev_pnt += 1
    while next_pnt < frames:
        if sum(ordered[next_pnt][0]) != 0:
            next_pnt += 1
            prev_pnt += 1
        else:
            prev_pnt -= 1

            while next_pnt < frames and sum(ordered[next_pnt][0]) == 0:
                next_pnt += 1

            start_data = None
            end_data = None

            if next_pnt != frames:
                start_data = ordered[prev_pnt]
                end_data = ordered[next_pnt]

            number_of_steps = next_pnt - prev_pnt

            for i in range(prev_pnt + 1, next_pnt):
                for k in range(0, num_points):
                    for l in range(0, 2):
                        step = 0
                        if next_pnt != frames:
                            step = (end_data[k][l] - start_data[k][l]) / float(number_of_steps)
                        ordered[i][k][l] = ordered[i - 1][k][l] + step

            prev_pnt = next_pnt

    return ordered

def interpolate_landmarks(ordered):
    # Only supports landmarks

    num_objects = 2
    num_points = 21

    for j in range(0, num_objects):
        prev_pnt = 0 # first location with data
        next_pnt = 0 # first next location with data
        frames = len(ordered)

        while next_pnt < frames and sum(ordered[next_pnt][j][0]) == 0:
            next_pnt += 1
            prev_pnt += 1
        while next_pnt < frames:
            if sum(ordered[next_pnt][j][0]) != 0:
                next_pnt += 1
                prev_pnt += 1
            else:
                prev_pnt -= 1

                while next_pnt < frames and sum(ordered[next_pnt][j][0]) == 0:
                    next_pnt += 1

                start_data = None
                end_data = None

                if next_pnt == frames:
                    continue

                if next_pnt != frames:
                    start_data = ordered[prev_pnt][j]
                    end_data = ordered[next_pnt][j]

                number_of_steps = next_pnt - prev_pnt

                for i in range(prev_pnt + 1, next_pnt):
                    for k in range(0, num_points):
                        for l in range(0, 2):
                            step = 0
                            if next_pnt != frames:
                                step = (end_data[k][l] - start_data[k][l]) / float(number_of_steps)
                            ordered[i][j][k][l] = ordered[i - 1][j][k][l] + step

                prev_pnt = next_pnt

    return ordered

def interpolate_hand(ordered_hands):
    for j in range(0, 2): # for both hands
 
        prev_pnt = 0 # first location with data
        next_pnt = 0 # first next location with data
        frames = len(ordered_hands)

        while next_pnt < frames and sum(ordered_hands[next_pnt][j]) == 0:
            next_pnt += 1
            prev_pnt += 1
        while next_pnt < frames:
            if sum(ordered_hands[next_pnt][j]) != 0:
                next_pnt += 1
                prev_pnt += 1
            else:
                prev_pnt -= 1

                while next_pnt < frames and sum(ordered_hands[next_pnt][j]) == 0:
                    next_pnt += 1

                # interpolate between (prev_pnt, next_pnt) if there was data at next_pnt
                if next_pnt != frames:
                    
                    start_data = ordered_hands[prev_pnt][j]
                    end_data = ordered_hands[next_pnt][j]

                    number_of_steps = next_pnt - prev_pnt

                    x1_step = (end_data[0] - start_data[0]) / float(number_of_steps)
                    y1_step = (end_data[1] - start_data[1]) / float(number_of_steps)
                    w1_step = (end_data[2] - start_data[2]) / float(number_of_steps)
                    h1_step = (end_data[3] - start_data[3]) / float(number_of_steps)

                    for i in range(prev_pnt + 1, next_pnt):
                        ordered_hands[i][j][0] = ordered_hands[i - 1][j][0] + x1_step 
                        ordered_hands[i][j][1] = ordered_hands[i - 1][j][1] + y1_step 
                        ordered_hands[i][j][2] = ordered_hands[i - 1][j][2] + w1_step 
                        ordered_hands[i][j][3] = ordered_hands[i - 1][j][3] + h1_step 
                else:
                    x1_step, y1_step, w1_step, h1_step = 0, 0, 0, 0
                    for i in range(prev_pnt + 1, next_pnt):
                        ordered_hands[i][j][0] = ordered_hands[i - 1][j][0] + x1_step 
                        ordered_hands[i][j][1] = ordered_hands[i - 1][j][1] + y1_step 
                        ordered_hands[i][j][2] = ordered_hands[i - 1][j][2] + w1_step 
                        ordered_hands[i][j][3] = ordered_hands[i - 1][j][3] + h1_step

                prev_pnt = next_pnt

    return ordered_hands

def assign_face_feature_id(data_file, feature_type):
    # Tracks a feature_type (faces) and returns an ordered array that best attempts to use overlap to minimize error.
    curr_data = load_json(data_file)
    if not curr_data:
        return None
    curr_data = {int(key): value for key, value in curr_data.items()}

    threshold = 0.015

    frames = len(curr_data)

    interpolated_data = [[[0 for i in range(2)] for j in range(6)] for k in range(frames)]

    start_assignment = False
    no_data_streak = 0
    last_assignment = -1

    for i in range(frames):

        num_faces = 0

        if curr_data[i]['faces'] != None:
            num_faces = len(curr_data[i]['faces'])

        # print(i)
        # print(curr_data[i]['faces'])
        
        if start_assignment:
            
            last_min_dist = 1e9
            last_is_assigned_to = None
            # find closest face
            for face in range(num_faces):
                new_dist = calculate_distance(curr_data[i]['faces']["{}".format(face)], interpolated_data[last_assignment], 'faces')
                if last_min_dist > new_dist:
                    last_min_dist = new_dist
                    last_is_assigned_to = curr_data[i]['faces']["{}".format(face)]

            if last_min_dist < iou_threshold(threshold, no_data_streak):
                no_data_streak = 0
                last_assignment = i
                for elem in range(0, 6):
                    interpolated_data[i][elem] = last_is_assigned_to['{}'.format(elem)]
            else:
                no_data_streak += 1

        if start_assignment == False and num_faces == 1:
            start_assignment = True
            last_assignment = i
            for elem in range(0, 6):
                interpolated_data[i][elem] = curr_data[i]['faces']["0"][str(elem)]

        # print(interpolated_data[i])
        # print()

    return interpolated_data


def assign_hand_feature_id(data_file, no_interpolate_3_bbox, feature_type):
    # Tracks a feature_type (boxes or landmarks) and returns an ordered array that best attempts to use overlap to minimize error. 


    curr_data = load_json(data_file)
    if not curr_data:
        return None
    curr_data = {int(key): value for key, value in curr_data.items()}

    threshold = 0.015

    frames = len(curr_data)
    interpolated_data = []
    if feature_type == 'landmarks':
        interpolated_data = [[[[0 for i in range(3)] for j in range(21)] for k in range(2)] for l in range(frames)]
    elif feature_type == 'boxes':
        interpolated_data = [[[0 for i in range(4)] for j in range(2)] for k in range(frames)]

    start_assignment = False

    last_0_no_data_streak = 0
    last_1_no_data_streak = 0

    last_0_assignment = -1
    last_1_assignment = -1
    
    for i in range(frames):

        curr_hand_0, curr_hand_1, curr_hand_2 = None, None, None

        if feature_type == 'landmarks' and curr_data[i]["landmarks"] != None:
            if len(curr_data[i]["landmarks"]) >= 1:
                curr_hand_0 = curr_data[i]["landmarks"]["0"]
            if len(curr_data[i]["landmarks"]) >= 2:
                curr_hand_1 = curr_data[i]["landmarks"]["1"]
            if len(curr_data[i]["landmarks"]) >= 3:
                curr_hand_2 = curr_data[i]["landmarks"]["2"]
    
        elif feature_type == 'boxes':
            if len(curr_data[i]["boxes"]) >= 1 and sum(curr_data[i]["boxes"]["0"][0:4]) != 0:
                curr_hand_0 = curr_data[i]["boxes"]["0"][0:4]
            if len(curr_data[i]["boxes"]) >= 2 and sum(curr_data[i]["boxes"]["1"][0:4]) != 0:
                curr_hand_1 = curr_data[i]["boxes"]["1"][0:4]
            if len(curr_data[i]["boxes"]) >= 3 and sum(curr_data[i]["boxes"]["2"][0:4]) != 0:
                curr_hand_2 = curr_data[i]["boxes"]["2"][0:4]


        # Set the earlier curr_hand
        if curr_hand_0 == None:
            if curr_hand_2 != None:
                curr_hand_0 = curr_hand_2
                curr_hand_2 = None
            elif curr_hand_1 != None:
                curr_hand_0 = curr_hand_1
                curr_hand_1 = None
        if curr_hand_1 == None:
            if curr_hand_2 != None:
                curr_hand_1 = curr_hand_2
                curr_hand_2 = None

        # print("\n{}".format(i))
        # print("{}\n{}\n{}".format(curr_hand_0, curr_hand_1, curr_hand_2))

        if start_assignment: # Note that at least one (last_center_0 or last_center_1 must be non-negative)
            last_0_is_assigned_to = None
            last_1_is_assigned_to = None
        
            if curr_hand_0 == None: # 0 bbox in this frame
                last_0_is_assigned_to = None
                last_1_is_assigned_to = None

            elif curr_hand_1 == None: # 1 bbox in this frame
                
                curr_center_0 = calculate_center(curr_hand_0, feature_type)

                dist_curr_center_0_to_last_center_0 = 1e9
                dist_curr_center_0_to_last_center_1 = 1e9

                if last_0_assignment != -1:
                    dist_curr_center_0_to_last_center_0 = calculate_distance(curr_center_0, calculate_center(interpolated_data[last_0_assignment][0], feature_type), feature_type)
                if last_1_assignment != -1:
                    dist_curr_center_0_to_last_center_1 = calculate_distance(curr_center_0, calculate_center(interpolated_data[last_1_assignment][1], feature_type), feature_type)

                if dist_curr_center_0_to_last_center_0 < dist_curr_center_0_to_last_center_1:
                    if dist_curr_center_0_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                        last_0_is_assigned_to = curr_hand_0
                else:
                    if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                        last_1_is_assigned_to = curr_hand_0

            elif curr_hand_2 == None: # 2 bbox in this frame
                # see which combination minimizes distance....
                curr_center_0 = calculate_center(curr_hand_0, feature_type)
                curr_center_1 = calculate_center(curr_hand_1, feature_type)

                last_0_is_assigned_to = None
                last_1_is_assigned_to = None

                if last_0_assignment != -1 and last_1_assignment != -1: # we have had data for both points before and currently, match minimize
                    last_center_0 = calculate_center(interpolated_data[last_0_assignment][0], feature_type)
                    last_center_1 = calculate_center(interpolated_data[last_1_assignment][1], feature_type)

                    # curr = option, last = hand
                    dist_curr_center_0_to_last_center_0 = calculate_distance(curr_center_0, last_center_0, feature_type)
                    dist_curr_center_1_to_last_center_0 = calculate_distance(curr_center_1, last_center_0, feature_type)

                    dist_curr_center_0_to_last_center_1 = calculate_distance(curr_center_0, last_center_1, feature_type)
                    dist_curr_center_1_to_last_center_1 = calculate_distance(curr_center_1, last_center_1, feature_type)

                    # NOTE: technically, dist might work better if it is shortest distance between point and threshold radius, but this should also work
                    # Maybe add a safety net -- If mediapipe measured some data but we did not add it (say 5 times), then our model may have desynced from it. However, this model should still scale and work.

                    if dist_curr_center_0_to_last_center_0 < dist_curr_center_1_to_last_center_0: # option_0 is better than option_1 for hand_0
                        # validate if option_0 is even good enough
                        if dist_curr_center_0_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = curr_hand_0
                        else:
                            last_0_is_assigned_to = None

                        if last_0_is_assigned_to == None: # no good option for hand_0, let's try to pair something with hand_1 instead
                            if dist_curr_center_0_to_last_center_1 < dist_curr_center_1_to_last_center_1: # option_0 is better than option_1 for hand_1
                                if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_0
                                    last_1_is_assigned_to = curr_hand_0
                            else: # option_1 is better than option_0 for hand_1
                                if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_1
                                    last_1_is_assigned_to = curr_hand_1

                        else: # hand_0 is assigned to option_0, but let's see what is better for hand_1...
                            if dist_curr_center_0_to_last_center_1 < dist_curr_center_1_to_last_center_1: # option_0 is better than option_1 for hand_1
                                if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                                    # option_0 is better and valid for both hands, so let's determine which hand option_1 is closer to, and see if it's possible to do this instead
                                    if dist_curr_center_1_to_last_center_0 < dist_curr_center_1_to_last_center_1: # option_1 is closer to hand_0 (c0)
                                        if dist_curr_center_1_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak): # validate option_1
                                            # this essentially says that even though option_0 is preferred for both, option_1 most likely belongs to hand_0
                                            last_0_is_assigned_to = curr_hand_1
                                            last_1_is_assigned_to = curr_hand_0

                                    else: # option_1 is closer to hand_1 (c1)
                                        if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                                            # this essentially says that even though option_0 is preferred for both, option_1 most likely belongs to hand_1
                                            last_0_is_assigned_to = curr_hand_0
                                            last_1_is_assigned_to = curr_hand_1

                                    if last_1_is_assigned_to == None: # option_1 was not good and thus last_1 is not assigned.
                                        # assign option_0 to closer one (valid for both)
                                        if dist_curr_center_0_to_last_center_0 < dist_curr_center_0_to_last_center_1:
                                            last_0_is_assigned_to = curr_hand_0
                                            last_1_is_assigned_to = None
                                        else:
                                            last_0_is_assigned_to = None
                                            last_1_is_assigned_to = curr_hand_0
                                else:
                                    last_1_is_assigned_to = None

                            else: # hand_0 is assigned to option_0, option_1 is better than option_0 for hand_1
                                if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_1
                                    last_1_is_assigned_to = curr_hand_1

                    else: # option_1 is better than option_0 for hand_0
                        # validate if option_1 is even good enough
                        if dist_curr_center_1_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = curr_hand_1
                        else:
                            last_0_is_assigned_to = None

                        if last_0_is_assigned_to == None: # no good option for hand_0, let's try to pair something with hand_1 instead
                            if dist_curr_center_0_to_last_center_1 < dist_curr_center_1_to_last_center_1: # option_0 is better than option_1 for hand_1
                                if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_0
                                    last_1_is_assigned_to = curr_hand_0
                            else: # option_1 is better than option_0 for hand_1
                                if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_1
                                    last_1_is_assigned_to = curr_hand_1

                        else: # hand_0 is assigned to option_1, but let's see what is better for hand_1...
                            if dist_curr_center_1_to_last_center_1 < dist_curr_center_0_to_last_center_1: # option_1 is better than option_0 for hand_1
                                if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                                    # option_1 is better and valid for both hands, so let's determine which hand option_0 is closer to, and see if it's possible to do this instead
                                    if dist_curr_center_0_to_last_center_0 < dist_curr_center_0_to_last_center_1: # option_0 is closer to hand_0 (c0)
                                        if dist_curr_center_0_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                                            # this essentially says that even though option_1 is preferred for both, option_0 most likely belongs to hand_0
                                            last_0_is_assigned_to = curr_hand_0
                                            last_1_is_assigned_to = curr_hand_1

                                    else: # option_0 is closer to hand_1 (c1)
                                        if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): 
                                            # this essentially says that even though option_1 is preferred for both, option_0 most likely belongs to hand_1
                                            last_0_is_assigned_to = curr_hand_1
                                            last_1_is_assigned_to = curr_hand_0

                                    if last_1_is_assigned_to == None: # option_0 was not good and thus last_1 is not assigned
                                        # assign option_1 to closer one (valid for both)
                                        if dist_curr_center_1_to_last_center_0 < dist_curr_center_1_to_last_center_1:
                                            last_0_is_assigned_to = curr_hand_1
                                            last_1_is_assigned_to = None
                                        else:
                                            last_0_is_assigned_to = None
                                            last_1_is_assigned_to = curr_hand_1
                                else:
                                    last_1_is_assigned_to = None

                            else: # hand_0 is assigned to option_1, option_0 is better than option_1 for hand_1
                                if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak): # validate option_0
                                    last_1_is_assigned_to = curr_hand_0

                    # print("last_0: ", last_0_is_assigned_to, last_0_assignment, last_0_no_data_streak, dist_curr_center_0_to_last_center_0, dist_curr_center_1_to_last_center_0, iou_threshold(threshold, last_0_no_data_streak))
                    # print("last_1: ", last_1_is_assigned_to, last_1_assignment, last_0_no_data_streak)


                elif last_0_assignment == -1 and last_1_assignment != -1: # only right hand data, no left hand data. Assign one that minimizes distance to last_center_1 to last_center_1/id=1, and other to last_center_0
                    last_center_1 = calculate_center(interpolated_data[last_1_assignment][1], feature_type)

                    dist_curr_center_0_to_last_center_1 = calculate_distance(curr_center_0, last_center_1, feature_type)
                    dist_curr_center_1_to_last_center_1 = calculate_distance(curr_center_1, last_center_1, feature_type)

                    if dist_curr_center_0_to_last_center_1 < dist_curr_center_1_to_last_center_1:
                        if dist_curr_center_0_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                            last_1_is_assigned_to = curr_hand_0
                            last_0_is_assigned_to = curr_hand_1
                    else:
                        if dist_curr_center_1_to_last_center_1 <= iou_threshold(threshold, last_1_no_data_streak):
                            last_1_is_assigned_to = curr_hand_1
                            last_0_is_assigned_to = curr_hand_0
                         

                elif last_0_assignment != -1 and last_1_assignment == -1:
                    last_center_0 = calculate_center(interpolated_data[last_0_assignment][0], feature_type)

                    dist_curr_center_0_to_last_center_0 = calculate_distance(curr_center_0, last_center_0, feature_type)
                    dist_curr_center_1_to_last_center_0 = calculate_distance(curr_center_1, last_center_0, feature_type)

                    if dist_curr_center_0_to_last_center_0 < dist_curr_center_1_to_last_center_0:
                        if dist_curr_center_0_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = curr_hand_0
                            last_1_is_assigned_to = curr_hand_1
                    else:
                        if dist_curr_center_1_to_last_center_0 <= iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = curr_hand_1
                            last_1_is_assigned_to = curr_hand_0

            else: # 3 (or more) bbox in this frame. Simple Assignment
                if no_interpolate_3_bbox:
                    continue
                curr_center_0 = calculate_center(curr_hand_0, feature_type)
                curr_center_1 = calculate_center(curr_hand_1, feature_type)
                curr_center_2 = calculate_center(curr_hand_2, feature_type)

                last_0_is_assigned_to = None
                last_1_is_assigned_to = None

                last_0_min_dist = None
                last_1_min_dist = None

                if last_0_assignment != -1:
                    last_center_0 = calculate_center(interpolated_data[last_0_assignment][0], feature_type)
                    dist_curr_center_0_to_last_center_0 = calculate_distance(curr_center_0, last_center_0, feature_type)
                    dist_curr_center_1_to_last_center_0 = calculate_distance(curr_center_1, last_center_0, feature_type)
                    dist_curr_center_2_to_last_center_0 = calculate_distance(curr_center_2, last_center_0, feature_type)

                    last_0_min_dist = dist_curr_center_0_to_last_center_0
                    last_0_is_assigned_to = curr_hand_0
                    if last_0_min_dist > dist_curr_center_1_to_last_center_0:
                        last_0_min_dist = dist_curr_center_1_to_last_center_0
                        last_0_is_assigned_to = curr_hand_1

                    if last_0_min_dist > dist_curr_center_2_to_last_center_0:
                        last_0_min_dist = dist_curr_center_2_to_last_center_0
                        last_0_is_assigned_to = curr_hand_2

                if last_1_assignment != -1:
                    last_center_1 = calculate_center(interpolated_data[last_1_assignment][1], feature_type)
                    dist_curr_center_0_to_last_center_1 = calculate_distance(curr_center_0, last_center_1, feature_type)
                    dist_curr_center_1_to_last_center_1 = calculate_distance(curr_center_1, last_center_1, feature_type)
                    dist_curr_center_2_to_last_center_1 = calculate_distance(curr_center_2, last_center_1, feature_type)

                    # print("{}\n{}\n{}".format(curr_center_0, curr_center_1, curr_center_2))
                    # print(last_center_1)

                    last_1_min_dist = dist_curr_center_0_to_last_center_1
                    last_1_is_assigned_to = curr_hand_0
                    if last_1_min_dist > dist_curr_center_1_to_last_center_1:
                        last_1_min_dist = dist_curr_center_1_to_last_center_1
                        last_1_is_assigned_to = curr_hand_1

                    if last_1_min_dist > dist_curr_center_2_to_last_center_1:
                        last_1_min_dist = dist_curr_center_2_to_last_center_1
                        last_1_is_assigned_to = curr_hand_2

                if last_0_is_assigned_to == last_1_is_assigned_to:
                    # let the closer one win.
                    if last_0_min_dist < last_1_min_dist:
                        last_1_is_assigned_to = None                        
                        if last_0_min_dist > iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = None
                    else:
                        last_0_is_assigned_to = None
                        if last_1_min_dist > iou_threshold(threshold, last_1_no_data_streak):
                            last_1_is_assigned_to = None
                else:
                    if last_0_is_assigned_to != None:
                        if last_0_min_dist > iou_threshold(threshold, last_0_no_data_streak):
                            last_0_is_assigned_to = None
                    if last_1_is_assigned_to != None:
                        # print(last_1_min_dist, iou_threshold(threshold, last_1_no_data_streak))
                        if last_1_min_dist > iou_threshold(threshold, last_1_no_data_streak):
                            last_1_is_assigned_to = None


            if last_0_is_assigned_to == None:
                last_0_no_data_streak += 1
            else:
                last_0_no_data_streak = 0
                last_0_assignment = i
                set_interpolated_data(interpolated_data, i, 0, last_0_is_assigned_to, feature_type)

            if last_1_is_assigned_to == None:
                last_1_no_data_streak += 1
            else:
                last_1_no_data_streak = 0
                last_1_assignment = i
                set_interpolated_data(interpolated_data, i, 1, last_1_is_assigned_to, feature_type)                

        if start_assignment == False and curr_hand_0 != None and curr_hand_2 == None: # Exactly one or two bbox returned on this frame
            start_assignment = True
            
            if curr_hand_1 == None:
                last_0_assignment = i
                set_interpolated_data(interpolated_data, i, 0, curr_hand_0, feature_type)

            else:
                if feature_type == 'boxes':
                    if curr_hand_0[0] <= 0.5:
                        interpolated_data[i][0] = curr_hand_0
                        last_0_assignment = i
                        if curr_hand_1[0] >= 0.5:
                            interpolated_data[i][1] = curr_hand_1
                            last_1_assignment = i
                    else:
                        interpolated_data[i][1] = curr_hand_0
                        last_1_assignment = i
                        if curr_hand_1[0] <= 0.5:
                            interpolated_data[i][0] = curr_hand_1
                            last_0_assignment = i     
                elif feature_type == 'landmarks':
                    # do stuff here
                    if curr_hand_0["0"][0] <= 0.5:
                        set_interpolated_data(interpolated_data, i, 0, curr_hand_0, feature_type)
                        last_0_assignment = i
                        if curr_hand_1["0"][0] >= 0.5:
                            set_interpolated_data(interpolated_data, i, 1, curr_hand_1, feature_type)
                            last_1_assignment = i
                    else:
                        set_interpolated_data(interpolated_data, i, 1, curr_hand_0, feature_type)
                        last_1_assignment = i
                        if curr_hand_1["0"][0] <= 0.5:
                            set_interpolated_data(interpolated_data, i, 0, curr_hand_1, feature_type)
                            last_0_assignment = i



        # print(interpolated_data[i][0])
        # print(interpolated_data[i][1])
        # print()

    return interpolated_data
