import os

import numpy as np
import pandas as pd

sample_rate = 50  # number of observation per second based on dataset documentation

sliding_size = int(.2 * sample_rate)  # number of skipped datapoints to start next window


############################################################################################ Feature sets


def mean_crossing_rate(col):
    # col = np.array(values)
    normalized = col - col.mean()  # to make elements of array possitive or negetive
    return ((normalized[:-1] * col[1:]) < 0).sum()  # Zero-Crossing_rate


def FS1(window):  # only mean

    avgs = list(window.mean()[:-1])

    label = window.iloc[:, -1].mode()[0]  ## select the most frequent label as the label of the window
    avgs.append(label)
    return avgs


def FS2(window):  # Mean and std

    features = []
    features.append(np.array(window.mean()[:-1]))
    features.append(np.array(window.std()[:-1]))
    features = np.hstack(features).tolist()

    label = window.iloc[:, -1].mode()[0]  ## select the most frequent label as the label of the window

    features.append(label)

    return features


def FS3(window):  # mean, std,max,min and zero-crossing-rate

    features = []
    features.append(np.array(window.mean()[:-1]))
    features.append(np.array(window.std()[:-1]))
    features.append(np.array(window.min()[:-1]))
    features.append(np.array(window.max()[:-1]))
    mean_crossing = [mean_crossing_rate(window.iloc[:, i].values) for i in range(window.shape[1] - 1)]
    features.append(np.array(mean_crossing))

    features = np.hstack(features).tolist()

    label = window.iloc[:, -1].mode()[0]  ## select the most frequent label as the label of the window
    features.append(label)
    return features


####################################################################################################################

def windowing_dataset(dataset, win_size, feature_extraction_function, subject_id, overlap=False):
    windowed_dataset = []
    win_count = 0
    if overlap:
        step_size = sliding_size  # for Overlapping technique
    else:
        step_size = win_size  # for Non-overlapping technique

    for index in range(0, dataset.shape[0], step_size):

        start = index
        end = start + win_size

        if (end <= dataset.shape[0]):  # to assure all of windows are equal in size
            window = dataset.iloc[start:end, :].reset_index(drop=True)
            win_count = win_count + 1
            features = feature_extraction_function(window)

            windowed_dataset.append(features)

    final = pd.DataFrame(windowed_dataset)
    final.insert(0, 'group', subject_id)  # to use in Subject CV
    return final


def Preprocessing(dataset_path, overlapping):
    features_functions = [FS1, FS2, FS3]
    win_sizes = np.linspace(.25, 7, 28, endpoint=True)
    dataframe = pd.read_csv(dataset_path,
                            usecols=['subject_ID', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',
                                     'activity_id'])
    for win_size in win_sizes:

        print("Start for win size {}".format(win_size))
        datapoints_per_window = int(win_size * sample_rate)

        for feature_function in features_functions:

            print(feature_function.__name__)

            transformed_db = windowing_dataset(dataframe, datapoints_per_window, feature_function, subject,
                                               overlap=overlapping)

            if overlapping:
                overlap_path = 'overlapping_windowed'
            else:
                overlap_path = 'non-overlapping_windowed'

            path_to_save = os.path.join(os.getcwd(), 'processed_dataset', overlap_path,
                                        feature_function.__name__,
                                        'windowed_{}.csv'.format(win_size))
            transformed_db.to_csv(path_to_save, index=False)

            print('Window size {} has been done!'.format(win_size))


############################################################################################################################

'''
 - Reads the raw data from input_path
 - Segments the raw datasets into windowed ones by different window sizes  
 - From each window it extracts FS1,FS2 and FS3.
 - Saves results in output_path. 

  Parameters:
    -----------
    dataset_path : Path of raw dataset

    output_path : Path to save the processed dataset

    overlapping : Controls the sliding windows technique;
    1: Overlapping sliding windows
    0: Non-overlapping sliding windows

'''

raw_dataset_path = os.path.join(os.getcwd(), 'raw_dataset')
path_to_save_data = os.path.join(os.getcwd(), 'processed_dataset')

dataset_path = os.path.join(os.getcwd(), 'raw_dataset', 'exercise_data.50.0000_singleonly_all_labels.csv')

Preprocessing(dataset_path=raw_dataset_path, output_path=path_to_save_data, overlapping=True)
