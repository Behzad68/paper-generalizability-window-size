import glob
import os
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.model_selection import (LeaveOneGroupOut, ShuffleSplit)

sns.set(style="whitegrid", palette="pastel", color_codes=True)
plt.rc('xtick', labelsize=15)
plt.rc('axes', labelsize=15)
plt.rc('axes', labelweight='bold')

plt.rcParams['font.weight'] = 'bold'
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['savefig.bbox'] = 'tight'
plt.rc('ytick', labelsize=15)
np.random.seed(1)


def plot_validation_figures():
    def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
        splits = cv.split(X=X, y=y, groups=group)

        for index, (training, test) in enumerate(splits):
            indices = np.array([np.nan] * len(X))
            indices[test] = 1
            indices[training] = 0

            ax.scatter(range(len(indices)), [index + .5] * len(indices),
                       c=indices, marker='_', lw=lw, cmap=plt.cm.coolwarm
                       )

        ax.scatter(range(len(X)), [index + 1.5] * len(X),
                   c=y, marker='_', lw=lw, cmap=plt.cm.tab20)

        ax.scatter(range(len(X)), [index + 2.5] * len(X),
                   c=group, marker='_', lw=lw, cmap=plt.cm.tab20)

        if (isinstance(cv, LeaveOneGroupOut)):
            n_splits = cv.get_n_splits(groups=group)

        yticklabels = list(range(n_splits)) + ['class', 'subject']
        ax.set(yticks=np.arange(n_splits + 2) + .5, yticklabels=yticklabels,
               xlabel='Sample index', ylabel="CV iteration",
               ylim=[n_splits + 2.2, -.2], xlim=[0, len(X)])

        return ax

    def plot_cv(dataset, CVs, n_splits):

        if (len(CVs) == 0):
            raise ValueError('There is any CV to plot.')

        dataset = pd.read_csv(dataset, sep='\t')

        groups = dataset['group'].values.ravel()

        X = dataset.iloc[:, 1:-1].values

        Y = dataset.iloc[:, dataset.shape[1] - 1].values

        project_root = os.path.dirname(os.path.dirname(__file__))
        output_folder = os.path.join(project_root, 'Figures')

        for cv in CVs:

            if (cv == LeaveOneGroupOut):
                cur_cv = cv()

            else:
                cur_cv = cv(n_splits=n_splits)

            fig_name = type(cur_cv).__name__
            fig, ax = plt.subplots(figsize=(8, 5))
            plot_cv_indices(cv=cur_cv, X=X, y=Y, group=groups, ax=ax, n_splits=n_splits)

            ax.legend([Patch(color='r'), Patch(color='b')],
                      ['Testing set', 'Training set'], loc=(1.02, .8))

            plt.tight_layout()
            fig.subplots_adjust(right=.7)

            plt.savefig('{}/{}.png'.format(output_folder, fig_name))

    dataset = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset1', 'processed_dataset', 'overlap',
                           'FS1',
                           'dataset0.5.csv')
    plot_cv(dataset=dataset, CVs=[LeaveOneGroupOut, ShuffleSplit], n_splits=10)
    return True


def plot_results_global():
    def filter_result(result_path, windowing_type):
        models = []
        f1_scores = []
        window_sizes = []
        classifiers = list(filter(lambda x: os.path.isdir(os.path.join(result_path, x)), os.listdir(result_path)))

        for cls in classifiers:

            files = glob.glob('{0}/*.csv'.format(os.path.join(result_path, cls)))
            for file in files:
                win_size = os.path.splitext(os.path.basename(file))[0].split(sep='_')[-1]
                rslt = pd.read_csv(file, index_col=0)
                rslt.index = rslt.index.str.strip()
                rslt.columns = rslt.columns.str.strip()
                f1_score = rslt.at['accuracy', 'f1-score']
                models.append(cls)
                window_sizes.append(win_size)
                f1_scores.append(f1_score)
        df = pd.DataFrame({'classifier': models, 'f1_score': f1_scores, 'win_size': window_sizes})
        df['window_type'] = windowing_type
        return df

    def plot_results(result_path, cv_type, dataset):
        for fs in ['FS1', 'FS2', 'FS3']:
            non_overlap_path = os.path.join(result_path, 'nonoverlap', fs, cv_type)
            overlap_path = os.path.join(result_path, 'overlap', fs, cv_type)

            overlap_df = filter_result(overlap_path, 'O')
            nooverlap_df = filter_result(non_overlap_path, 'NO')
            mix_df = overlap_df.append(nooverlap_df).sort_values(['classifier', 'win_size'])

            fig, ax = plt.subplots()
            ax = sns.violinplot(ax=ax, x="classifier", y="f1_score", hue='window_type',
                                split=True,
                                palette="Set2", hue_order=["NO", 'O'], scale='width',
                                data=mix_df, inner=None)
            ax.set(ylim=(0, 1))
            plt.legend()

            path_to_save = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures',
                                        '{}_{}_{}.png'.format(dataset, cv_type, fs))
            plt.savefig(path_to_save, format='png')
            plt.close()

    result_path_dataset1 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset1/results')
    result_path_dataset2 = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Dataset2/results')

    for cv in ['iid', 'sbj']:
        plot_results(result_path=result_path_dataset1, cv_type=cv, dataset='Dataset1')
        plot_results(result_path=result_path_dataset1, cv_type=cv, dataset='Dataset1')
        plot_results(result_path=result_path_dataset2, cv_type=cv, dataset='Dataset2')
        plot_results(result_path=result_path_dataset2, cv_type=cv, dataset='Dataset2')
    return True


def plot_results_per_activity():
    dataset1_activities = {'No activity': 0, 'Walking': 1, 'Jogging': 2, 'Running': 3, 'Jump up': 4,
                           'Jump front and back': 5,
                           'Jump sideways': 6, 'Jump leg/arms open/closed': 7, 'Jump rope': 8
        , 'Trunk twist (arms outstretched)': 9,
                           'Trunk twist (elbows bent)': 10, 'Waist bends forward': 11, 'Waist rotation': 12,
                           'Waist bends (reach foot with opposite hand)': 13, 'Reach heels backwards': 14
        , 'Lateral bend': 15, 'Lateral bend with arm up': 16, 'Repetitive forward stretching': 17,
                           'Upper trunk and lower body opposite twist': 18, 'Lateral elevation of arms': 19,
                           'Frontal elevation of arms': 20
        , 'Frontal hand claps': 21, 'Frontal crossing of arms': 22, 'Shoulders high-amplitude rotation': 23,
                           'Shoulders low-amplitude rotation': 24, 'Arms inner rotation': 25,
                           'Knees (alternating) to the breast': 26,
                           'Heels (alternatively) to the backside': 27, 'Knees bending (crouching)': 28,
                           'Knees (alternating) bending forward': 29, 'Rotation on the knees': 30, 'Rowing': 31,
                           'Elliptical bike': 32,
                           'Cycling': 33}

    dataset2_activities = {'Noise': 1, 'Band Pull-Down Row': 2, 'Bicep Curl': 3, 'Box Jump (on bench)': 4, 'Burpee': 5,
                           'Butterfly sit-up': 6, 'Chest Press': 7, 'Crunch': 8,
                           'Dip': 9,
                           'Dumbbell Deadlift Row': 10, 'Dumbbell Row (both)': 11,
                           'Dumbbell Row (right)': 12, 'Dumbbell Squat (hands at side)': 13, 'Elliptical machine': 14,
                           'Punches': 15,
                           'Jump Rope': 16, 'Jumping Jacks': 17, 'Kettlebell Swing': 18, 'Lateral Raise': 19,
                           'Lawnmower (both)': 20,
                           'Lawnmower (right)': 21, 'Lunge (both legs)': 22
        , 'Ball Slam': 23, 'Triceps Extension (standing - both)': 24, 'Plank': 25,
                           'Power Boat pose': 26, 'Pushups': 27, 'Rowing machine': 28, 'Running': 29,
                           'Russian Twist': 30
        , 'Seated Back Fly': 31, 'Shoulder Press': 32, 'Squat': 33, 'Triceps Kickback': 34,
                           'Triceps extension (lying)': 35,
                           'Dumbbell Curl': 36, 'V-up': 37, 'Walk': 38, 'Walking lunge': 39, 'Wall Ball': 40,
                           'Wall Squat': 41}

    def _add_strategy(df_path, stg, activities):
        win_size = os.path.splitext(os.path.basename(df_path))[0].split(sep='_')[-1]
        df = pd.read_csv(df_path, skipfooter=3, engine='python')
        df.columns = df.columns.str.strip()
        df.drop(labels=['precision', 'recall', 'support'], axis=1, inplace=True)
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df.columns = ['Activity', 'F1-score']
        df = df[~df['Activity'].isin(['No activity', 'Noise'])]
        df['Activity'] = df['Activity'].map(activities)
        df['strategy'] = stg
        df['win_size'] = float(win_size)
        return df

    def plot_per_activity(model, dataset, fs='FS3', cv='sbj'):
        result_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '{}/results'.format(dataset))

        overlap_path = os.path.join(result_path, 'overlap', fs, cv, model)
        non_overlap_path = os.path.join(result_path, 'nonoverlap', fs, cv, model)

        results_O = glob.glob(os.path.join(overlap_path, '*.csv'))
        results_NO = glob.glob(os.path.join(non_overlap_path, '*.csv'))
        activities = dataset1_activities
        if dataset == 'Dataset2':
            activities = dataset2_activities
        dfs_O = list(map(lambda path: _add_strategy(path, 'O', activities), results_O))
        dfs_NO = list(map(lambda path: _add_strategy(path, 'NO', activities), results_NO))

        mix_df = reduce(lambda left, right: pd.concat([left, right], axis=0), [*dfs_NO, *dfs_O])
        mix_df.sort_values(by=['Activity', 'win_size'], inplace=True)

        fig, ax = plt.subplots()

        ax = sns.violinplot(ax=ax, x="Activity", y="F1-score", hue='strategy',
                            split=True, inner=None, scale='width',
                            palette="Set2",
                            data=mix_df)
        ax.set_ylim(0, 1)
        sns.despine(left=True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 1))
        path_to_save = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Figures',
                                    'per_activity_{}_{}_{}_{}.png'.format(dataset, model, cv, fs))
        plt.savefig(path_to_save, format='png')

    models = ['KNN', 'DT', 'NB', 'NCC']

    for model in models:
        plot_per_activity(model=model, fs='FS3', cv='sbj', dataset='Dataset1')
        plot_per_activity(model=model, fs='FS3', cv='sbj', dataset='Dataset2')
    return True


def test_plot_figures():
    # assert plot_validation_figures()
    # assert plot_results_global()
    assert plot_results_per_activity()
