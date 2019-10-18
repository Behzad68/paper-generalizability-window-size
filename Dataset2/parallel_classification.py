import concurrent
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

activities = ['Noise', 'Band Pull-Down Row', 'Bicep Curl (-+band)', 'Box Jump (on bench)', 'Burpee',
              'Butterfly Sit-up_Sit-up (hands positioned behind head)_Sit-ups', 'Chest Press (rack)', 'Crunch', 'Dip',
              'Dumbbell Deadlift Row', 'Dumbbell Row (knee on bench) (label spans both arms)',
              'Dumbbell Row (knee on bench) (right arm)', 'Dumbbell Squat (hands at side)', 'Elliptical machine',
              'Fast Alternating Punches',
              'Jump Rope', 'Jumping Jacks', 'Kettlebell Swing', 'Lateral Raise', 'Lawnmower (label spans both arms)',
              'Lawnmower (right arm)', 'Lunge (alternating both legs, weight optional)'
    , 'Medicine Ball Slam', 'Overhead Triceps Extension(+-abel spans both arms)', 'Plank(Left_Right side)',
              'Power Boat pose', 'Pushups', 'Rowing machine', 'Running (treadmill)',
              'Russian Twist'
    , 'Seated Back Fly', 'Shoulder Press (dumbbell)', 'Squat', 'Triceps Kickback', 'Triceps extension (lying down)',
              'Two-arm Dumbbell Curl', 'V-up', 'Walk', 'Walking lunge', 'Wall Ball', 'Wall Squat']

models = {'DT': DecisionTreeClassifier(criterion='entropy'), 'NB': GaussianNB(),
          'NCC': NearestCentroid(), "KNN": KNeighborsClassifier(n_neighbors=3, n_jobs=-1)}


def per_class_classification(file, cv='iid', overlap=True):
    if overlap:
        overlap_path = 'overlap'
    else:
        overlap_path = 'nonoverlap'

    file_name = os.path.basename(os.path.splitext(file)[0])
    fs = os.path.basename(os.path.dirname(file))
    win_size = file_name[file_name.index("_") + 1:]
    dataset = pd.read_csv(file)
    groups = dataset.iloc[:, 0]
    X = dataset.iloc[:, 1:-1].values
    Y = dataset.iloc[:, - 1].values
    for model_name, model in models.items():
        print(model_name)
        y_pred_overall = None
        y_test_overall = None
        if cv == 'sbj':
            cv_path = 'sbj'

            logo = LeaveOneGroupOut()
            for train_index, test_index in logo.split(X, Y, groups=groups):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                classifier = model
                classifier.fit(x_train, y_train)

                y_pred = classifier.predict(x_test)

                if y_pred_overall is None:
                    y_pred_overall = y_pred
                    y_test_overall = y_test
                else:
                    y_pred_overall = np.concatenate([y_pred_overall, y_pred])
                    y_test_overall = np.concatenate([y_test_overall, y_test])

        else:

            cv_path = 'iid'
            cv = KFold(n_splits=10, shuffle=True, random_state=1)
            for train_index, test_index in cv.split(X, Y):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                classifier = model
                classifier.fit(x_train, y_train)

                y_pred = classifier.predict(x_test)

                if y_pred_overall is None:
                    y_pred_overall = y_pred
                    y_test_overall = y_test
                else:
                    y_pred_overall = np.concatenate([y_pred_overall, y_pred])
                    y_test_overall = np.concatenate([y_test_overall, y_test])

        report = classification_report(y_test_overall, y_pred_overall, output_dict=True, digits=2
                                       , target_names=activities)

        output_path = os.path.join(os.getcwd(), 'results', overlap_path, fs, cv_path, model_name,
                                   '{}_{}.csv'.format(model_name, win_size))

        pd.DataFrame(report).transpose().to_csv(output_path, sep=',', index=False)


# per_class_classification(file=files[0])
dataset_path = os.path.join(os.getcwd(), 'processed_dataset', 'non_overlap')
feature_sets = ['FS1', 'FS2', 'FS3']
for fs in feature_sets:
    print(fs)
    dataset_path_fs = os.path.join(dataset_path, fs)
    files = glob.glob('{0}/*.csv'.format(dataset_path_fs))

    with concurrent.futures.ProcessPoolExecutor() as executer:
        executer.map(per_class_classification, files)
