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

activities = ['No activity', 'Walking', 'Jogging', 'Running', ' Jump up', 'Jump front and back',
              'Jump sideways', 'Jump leg/arms open/closed ', 'Jump rope'
    , 'Trunk twist (arms outstretched)',
              'Trunk twist (elbows bent)', 'Waist bends forward', ' Waist rotation',
              'Waist bends (reach foot with opposite hand)', 'Reach heels backwards'
    , 'Lateral bend', 'Lateral bend with arm up', 'Repetitive forward stretching',
              'Upper trunk and lower body opposite twist', 'Lateral elevation of arms',
              'Frontal elevation of arms'
    , 'Frontal hand claps', 'Frontal crossing of arms', 'Shoulders high-amplitude rotation',
              'Shoulders low-amplitude rotation', 'Arms inner rotation',
              'Knees (alternating) to the breast',
              'Heels (alternatively) to the backside', 'Knees bending (crouching)',
              'Knees (alternating) bending forward', 'Rotation on the knees', 'Rowing', 'Elliptical bike',
              'Cycling']

models = {'DT': DecisionTreeClassifier(criterion='entropy'), 'NB': GaussianNB(),
          'NCC': NearestCentroid(), "KNN": KNeighborsClassifier(n_neighbors=3)}


def per_class_classification(file, cv_type='iid', overlap=False):
    if overlap:
        overlap_path = 'overlap'
    else:
        overlap_path = 'nonoverlap'

    file_name = os.path.basename(os.path.splitext(file)[0])
    fs = os.path.basename(os.path.dirname(file))
    print(fs)
    win_size = (file_name[7:])
    print(str(win_size))
    dataset = pd.read_csv(file, sep='\t')
    groups = dataset.iloc[:, 1]
    X = dataset.iloc[:, 2:-1].values
    Y = dataset.iloc[:, - 1].values
    for model_name, model in models.items():
        print(model_name)
        y_pred_overall = None
        y_test_overall = None
        if cv_type == 'sbj':

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

        report = classification_report(y_test_overall, y_pred_overall, output_dict=True, digits=2,
                                       target_names=activities)

        output_path = os.path.join(os.getcwd(), 'results', overlap_path, fs, cv_path, model_name,
                                   '{}_{}.csv'.format(model_name, win_size))

        pd.DataFrame(report).transpose().to_csv(output_path, sep=',')


dataset_path = os.path.join(os.getcwd(), 'processed_dataset', 'non_overlap')
feature_sets = ['FS1', 'FS2', 'FS3']
for fs in feature_sets:
    dataset_path_fs = os.path.join(dataset_path, fs)
    files = glob.glob('{0}/*.csv'.format(dataset_path_fs))
    with concurrent.futures.ProcessPoolExecutor() as executer:
        rs = executer.map(per_class_classification, files)
        print(list(rs))
