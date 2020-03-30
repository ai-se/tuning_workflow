from __future__ import print_function, division
# import os
# cwd = os.getcwd()
# import_path=os.path.abspath(os.path.join(cwd, '..'))
# from demo import *
import sys
sys.path.append("models/Flash/")
from random import seed
from data_handler import read_data
from models.Flash.tuner import *
from models.Flash.flash import *
from sklearn.metrics import confusion_matrix


import pickle

metrics = ["g", "f1"]
learners = ['dt', 'rf']

def execute(res=''):
    seed(47)
    np.random.seed(47)
    datasets = read_data()

    dic={}
    l=[]

    for atype in datasets.keys():
        print(atype)
        dic[atype] = {}
        df = datasets[atype]
        #import pdb
        #pdb.set_trace()
        X, y = df[df.columns[:-1]], df[df.columns[-1]]
        sss = StratifiedShuffleSplit(n_splits=5, test_size=.2)
        for m in metrics:
            print(m)
            dic[atype][m] = {}
            for l in learners:
                dic[atype][m][l] = {'flash': [], 'default': []}
                for train_index, test_index in sss.split(X, y):
                    train_df = df.iloc[train_index]
                    test_df = df.iloc[test_index]
                    tuner = get_tuner(l)
                    best_config = tuning(tuner, train_df, project_name="", metric=m)
                    default_config = tuner.default_config

                    x_train, y_train = train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]]
                    x_test, y_test = test_df[test_df.columns[:-1]], test_df[test_df.columns[-1]]
                    tuned_score = measure_fitness(tuner, x_train, y_train, x_test, y_test, best_config, m)
                    default_score = measure_fitness(tuner, x_train, y_train, x_test, y_test, default_config, m)
                    dic[atype][m][l]['flash'].append(tuned_score)
                    dic[atype][m][l]['default'].append(default_score)
                print(l, dic[atype][m][l])
            print()
        print("*"*10)

    with open('dump/flash.pickle', 'wb') as handle:
        pickle.dump(dic, handle)


def get_features_importance():
    seed(47)
    np.random.seed(47)
    datasets = read_data('multiclass')

    dic = {}
    l = []

    for atype in datasets.keys():
        print(atype)
        dic[atype] = {}
        df = datasets[atype]
        X, y = df[df.columns[:-1]], df[df.columns[-1]]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
        for l in [learners[1]]:
            for train_index, test_index in sss.split(X, y):
                train_df = df.iloc[train_index]
                test_df = df.iloc[test_index]
                tuner = get_tuner(l)
                default_config = tuner.default_config
                clf = tuner.get_clf(default_config)
                x_train, y_train = train_df[train_df.columns[:-1]], train_df[train_df.columns[-1]]
                x_test, y_test = test_df[test_df.columns[:-1]], test_df[test_df.columns[-1]]
                clf.fit(x_train, y_train)
                prediction = clf.predict(x_test)
                cm = confusion_matrix(y_test, prediction)

                key_feats_indices = np.argsort(clf.feature_importances_)[::-1][:5]
                for index in key_feats_indices:
                    print("%s: %s" % (df.columns[index], clf.feature_importances_[index]), end="; ")
                import pdb
                pdb.set_trace()
            print()




if __name__ == '__main__':
    #eval(cmd())
    get_features_importance()
    #execute()