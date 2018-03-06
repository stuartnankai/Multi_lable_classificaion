import logging
import os

import errno
import pandas as pd
from random import randint
import numpy as np
from bs4 import BeautifulSoup
from langdetect import *
from langdetect import lang_detect_exception
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
from translate import Translator
from nltk import word_tokenize, re, RegexpTokenizer
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import os, shutil

# Load NLTK's English stop-words list
stop_words = set(stopwords.words('english'))

from magpie import Magpie


def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx + 1)
            indices.append(idx)
        except ValueError:
            break
    return indices


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words)


def grid_search(train_x, train_y, test_x, usemodel, genres, parameters, pipeline):
    results = {}
    if usemodel:
        with open('best_model_svm_1', "rb") as fp:
            best_clf = pickle.load(fp)
    else:

        grid_search_tune = GridSearchCV(pipeline, parameters, cv=5, n_jobs=5, verbose=30)
        grid_search_tune.fit(train_x, train_y)
        print("Best parameters set:")
        print(grid_search_tune.best_estimator_.steps)
        # measuring performance on test set
        print("Applying best classifier on test data:")
        best_clf = grid_search_tune.best_estimator_
        if not os.path.exists("best_model_svm"):
            with open('best_model_svm', "wb") as fp:
                pickle.dump(best_clf, fp)

    predictions = best_clf.predict(test_x)

    for n in range(len(predictions)):
        position_list = [genres[m] for m in [n for n, x in enumerate(predictions[n]) if x == 1]]

        results[test_x[n]] = position_list

    return results


def prepare_file(df):
    group_list = []

    for i in range(len(df)):
        x = ast.literal_eval(df.loc[i, 'group_id'])
        for j in x:
            group_list.append(j)
    group_set_list = list(set(group_list))

    for i in group_set_list:
        df[i] = 0

    for i in range(len(df)):
        x = ast.literal_eval(df.loc[i, 'group_id'])
        same_value = set(x) & set(group_set_list)
        same_value_list = [x for x in iter(same_value)]
        for j in same_value:
            df.loc[i, j] = 1

    df.to_csv('train.csv',
              sep=',', encoding='utf-8',
              index=False)
    return pd.read_csv('train.csv', sep=',')


def SVM(bigdata, x_test, target, usemodel):
    choices_list = ['nb', 'linearSVC', 'logit']
    choice_num = 1

    data_df_train = prepare_file(bigdata)

    print("Loading already processed training data....................")

    # data_df_train = pd.read_csv('Occup_group/cleaned_label_17000.csv', sep=',')
    #
    # # all the list of genres to be used by the classification report
    occupation = list(data_df_train.drop(['title', 'group_id'], axis=1).columns.values)  # label

    x_train = data_df_train[[target]].as_matrix()
    y_train = data_df_train.drop(['title', 'group_id'], axis=1).as_matrix()
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25)

    # x_test = data_df_train[[target]].as_matrix()

    # # transform matrix of description into lists to pass to a TfidfVectorizer
    train_x = [x[0].strip() for x in x_train.tolist()]
    test_x = [x[0].strip() for x in x_test.tolist()]

    #
    if choices_list[choice_num] == 'nb':
        # MultinomialNB: Multi-Class OneVsRestClassifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__estimator__alpha': (1e-2, 1e-3)
        }
        grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline)
        # exit(-1)
        print('Naive Bayes DONE>>>>>>>>>>>>>>>>>>>>')

    if choices_list[choice_num] == 'linearSVC':
        # LinearSVC
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            "clf__estimator__C": [0.01, 0.1, 1],
            "clf__estimator__class_weight": ['balanced', None],
        }
        print('SVM linear DONE>>>>>>>>>>>>>>>>>>>>')
        return grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline=pipeline)
        # exit(-1)
    #
    if choices_list[choice_num] == 'logit':
        # LogisticRegression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            "clf__estimator__C": [0.01, 0.1, 1],
            "clf__estimator__class_weight": ['balanced', None],
        }
        grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline)
        print('Logistic Regression DONE>>>>>>>>>>>>>>>>>>>>')
        # exit(-1)


def SVM_best(df, target, usemodel):
    choices_list = ['nb', 'linearSVC', 'logit']
    choice_num = 1
    x_test = df[[target]].as_matrix()
    print("Loading already processed training data....................")

    data_df_train = pd.read_csv('Occup_group/cleaned_label_17000.csv', sep=',')
    #
    # # all the list of genres to be used by the classification report
    occupation = list(data_df_train.drop(['title', 'id', 'description', 'group_id'], axis=1).columns.values)  # label
    #     #
    # split the data, leave 1/4 out for testing
    x_train = data_df_train[[target]].as_matrix()
    y_train = data_df_train.drop(['title', 'id', 'description', 'group_id'], axis=1).as_matrix()
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25)
    # transform matrix of description into lists to pass to a TfidfVectorizer
    train_x = [x[0].strip() for x in x_train.tolist()]
    test_x = [x[0].strip() for x in x_test.tolist()]

    if choices_list[choice_num] == 'nb':
        # MultinomialNB: Multi-Class OneVsRestClassifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(MultinomialNB(
                fit_prior=True, class_prior=None))),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'clf__estimator__alpha': (1e-2, 1e-3)
        }
        grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline)
        # exit(-1)
        print('Naive Bayes DONE>>>>>>>>>>>>>>>>>>>>')

    if choices_list[choice_num] == 'linearSVC':
        # LinearSVC
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            "clf__estimator__C": [0.01, 0.1, 1],
            "clf__estimator__class_weight": ['balanced', None],
        }
        print('SVM linear DONE>>>>>>>>>>>>>>>>>>>>')
        return grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline=pipeline)
        # exit(-1)
    #
    if choices_list[choice_num] == 'logit':
        # LogisticRegression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
        ])
        parameters = {
            'tfidf__max_df': (0.25, 0.5, 0.75),
            'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
            "clf__estimator__C": [0.01, 0.1, 1],
            "clf__estimator__class_weight": ['balanced', None],
        }
        grid_search(train_x, y_train, test_x, usemodel, occupation, parameters, pipeline)
        print('Logistic Regression DONE>>>>>>>>>>>>>>>>>>>>')
        # exit(-1)


def Deep_learning(df, x_test, target):
    folder = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    folder = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories/'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

    lab_list = []
    for i, row in df.iterrows():
        if i > len(df):
            break
        else:
            file_name = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories/' + str(i) + '.txt'
            lab_name = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories/' + str(i) + '.lab'

            title_data = df.at[i, target].encode('ascii', 'ignore').decode('ascii')

            with open(file_name, 'w') as the_file:
                the_file.write(title_data)

            row_data = eval(df.at[i, 'group_id'])
            for j in row_data:
                lab_list.append(j)
                with open(lab_name, 'a') as the_file:
                    the_file.write(str(j) + '\n')
    lab_set = list(set(lab_list))
    file = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/' + 'categories' + '.labels'
    for i in lab_set:
        with open(file, 'a') as the_file:
            the_file.write(str(i) + '\n')

    magpie = Magpie()
    # magpie.train_word2vec('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories', vec_dim=100)
    # magpie.fit_scaler('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories')

    magpie.init_word_vectors('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories', vec_dim=100)

    with open('test_data/categories.labels') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]
    magpie.train('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/test_data/categories', labels, test_ratio=0.0,
                 epochs=20)

    results_dl = {}

    df_test = pd.DataFrame(np.atleast_2d(x_test), columns=['title'])

    for i, row in df_test.iterrows():
        title_data = df_test.at[i, target].encode('ascii', 'ignore').decode('ascii')
        title_data = preprocess(title_data)
        # print("This is title: ", title_data)
        df_test.at[i, target] = title_data

        pre_label = [s[0] for s in magpie.predict_from_text(title_data) if s[1] >= 0.25]
        # print("This is test: ", title_data)
        # print("This is predict label: ", pre_label)
        results_dl[title_data] = pre_label
    return results_dl


def get_score(results_svm, results_dl, real_result):
    potints_svm = 0
    points_dl = 0
    total_points = 0

    for title, label in real_result.items():
        # print("This is svm: ",  )
        bonus_svm = len([i for i in label if i in results_svm.get(title)])  # if hit the label, get 1 point
        bonus_dl = len([i for i in label if i in results_dl.get(title)])

        punish_svm = len([i for i in label if i not in results_svm.get(title)])  # if mismatch the label, get -1 point
        punish_dl = len([i for i in label if i not in results_dl.get(title)])

        # print("This is punish svm: ", punish_svm)
        # print("This is punish dl: ", punish_dl)

        potints_svm += bonus_svm
        # potints_svm -= punish_svm
        points_dl += bonus_dl
        # points_dl -= punish_dl
        total_points += len(label)
        # print("This is svm: ", [i for i in label if i in results_svm.get(title)])
        # print("This is dl: ", [i for i in label if i in results_dl.get(title)])

    print("This is points for svm: ", potints_svm)
    print("This is points for DL: ", points_dl)
    print("This is total points: ", total_points)


#
# if __name__ == '__main__':
#     df = pd.read_csv('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/Occup_group/cleaned_17000.csv')
#     target = 'title'
#     data_x = df[[target]].as_matrix()
#     data_y = df.drop(['title', 'id', 'description'], axis=1).as_matrix()
#
#     x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.20)
#
#     bigdata_1 = pd.DataFrame(np.atleast_2d(x_train), columns=['title'])
#     bigdata_2 = pd.DataFrame(np.atleast_2d(y_train), columns=['group_id'])
#
#     bigdata = pd.concat([bigdata_1, bigdata_2], axis=1)
#
#     results_dl = Deep_learning(bigdata,x_test,target)
#     results_svm = SVM(bigdata, x_test, target=target, usemodel=False)
#     # print("This is svm: ", results_svm)
#     # print("This is dl: ", results_dl )
#
#     real_result = {}
#
#     for i in range(len(x_test)):
#         y = ast.literal_eval(y_test[i][0])
#         y = [n for n in y]
#         real_result[x_test[i][0]] = y
#
#     print("This is svm: ", results_svm)
#     print("This is dl: ", results_dl)
#     print("This is real: ", real_result)
#
#     get_score(results_svm,results_dl,real_result)
#

"""
Prepare corpus of labeled data 
"""
df = pd.read_csv('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/Occup_group/cleaned_18000.csv')
df = df.drop(['id'], axis=1)
lab_list = []
for i, row in df.iterrows():
    if i > len(df):
        break
    else:
        file_name = '/Users/sunxuan/Documents/Impactpool/seniority analysis/googlecloud_magpie/data/categories/' + str(i) + '.txt'
        lab_name = '/Users/sunxuan/Documents/Impactpool/seniority analysis/googlecloud_magpie/data/categories/' + str(i) + '.lab'

        title_data = df.at[i, 'title'].encode('ascii', 'ignore').decode('ascii')

        with open(file_name, 'w') as the_file:
            the_file.write(title_data)

        row_data = eval(df.at[i, 'group_id'])
        for j in row_data:
            lab_list.append(j)
            with open(lab_name,'a') as the_file:
                the_file.write(str(j)+'\n')
lab_set = list(set(lab_list))
file = '/Users/sunxuan/Documents/Impactpool/seniority analysis/googlecloud_magpie/data/' + 'categories' + '.labels'
for i in lab_set:
    with open(file, 'a') as the_file:
        the_file.write(str(i) + '\n')


"""
train process
"""
magpie = Magpie()
# magpie.train_word2vec('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', vec_dim=100)
# magpie.fit_scaler('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories')

magpie.init_word_vectors('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', vec_dim=100)

with open('data/categories.labels') as f:
    labels = f.readlines()
labels = [x.strip() for x in labels]
magpie.train('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', labels, test_ratio=0.0, epochs=30)

# """
# Save model
# """
#
# magpie.save_word2vec_model('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/embeddings/here')
# magpie.save_scaler('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/scaler/here', overwrite=True)
# magpie.save_model('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/model/here.h5')


"""
Reinitialize
"""

# with open('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories.labels') as f:
#     labels = f.readlines()
# labels = [x.strip() for x in labels]
#
# magpie = Magpie(
#     keras_model='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/model/here.h5',
#     word2vec_model='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/embeddings/here',
#     scaler='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/scaler/here',
#     labels=labels
# )

"""
Prediction
"""
# df = pd.read_csv('/Users/sunxuan/Documents/Impactpool/find_no_occup_job_20180209.csv')
# df = df.drop(['id', 'description'], axis=1)
# print("This is df: ", df.head())
# results_dl = {}
#
# for i, row in df.iterrows():
#     title_data = df.at[i, 'title'].encode('ascii', 'ignore').decode('ascii')
#     title_data = preprocess(title_data)
#     # print("This is title: ", title_data)
#     df.at[i, 'title'] = title_data
#
#     pre_label = [s[0] for s in magpie.predict_from_text(title_data) if s[1] >= 0.2]
#     print("This is title: ", title_data)
#     print("This is predict label: ", pre_label)
#     results_dl[title_data] = pre_label
#
# results_svm = SVM_best(df, target='title', usemodel=True)
#
# print("This is SVM: ", results_svm)
# print("This is DL: ", results_dl)

"""
Group test
# """
# total_svm = 0
# total_dl = 0
# total = 0
#
# for i in range(0, 100):
#     x = [randint(0, 3509) for p in range(0, 800)]  # Random pick titles as test data
#     results_dl = {}
#     real_result = {}
#     df = pd.DataFrame(data=None)
#     for i in range(len(x)):
#         lab_name = 'data/categories/' + str(x[i]) + '.lab'
#         file_name = 'data/categories/' + str(x[i]) + '.txt'
#         with open(lab_name) as f:
#             test_lab = f.readlines()
#         with open(file_name) as fb:
#             title_name = fb.readlines()
#             title_name = re.sub("[^a-zA-Z]",  # Search for all non-letters
#                                 " ",  # Replace all non-letters with spaces
#                                 str(title_name))
#         test_lab = [x.strip() for x in test_lab]
#         title_name = title_name.strip()
#         pre_label = [s[0] for s in magpie.predict_from_text(title_name) if s[1] >= 0.05]
#         # print("This is predict label: ", pre_label)
#         # print("This is real label: ", test_lab)
#         results_dl[title_name] = pre_label
#         real_result[title_name] = test_lab
#         df.at[i, 'title'] = title_name
#
#     print('Deep learning DONE>>>>>>>>>>>>>>>>>>>>')
#
#     results_svm = SVM_best(df, target='title', usemodel=True)
#
#     # print("This is svm: ", results_svm)
#     # print("This is dl: ", results_dl)
#     # print("This is real : ", real_result)
#     potints_svm = 0
#     points_dl = 0
#     total_points = 0
#
#     for title, label in real_result.items():
#         # print("This is svm: ",  )
#         bonus_svm = len([i for i in label if i in results_svm.get(title)])  # if hit the label, get 1 point
#         bonus_dl = len([i for i in label if i in results_dl.get(title)])
#
#         punish_svm = len([i for i in label if i not in results_svm.get(title)])  # if mismatch the label, get -1 point
#         punish_dl = len([i for i in label if i not in results_dl.get(title)])
#
#         # print("This is punish svm: ", punish_svm)
#         # print("This is punish dl: ", punish_dl)
#
#         potints_svm += bonus_svm
#         potints_svm -= punish_svm
#         points_dl += bonus_dl
#         points_dl -= punish_dl
#         total_points += len(label)
#         # print("This is svm: ", [i for i in label if i in results_svm.get(title)])
#         # print("This is dl: ", [i for i in label if i in results_dl.get(title)])
#
#     # print("This is points for svm: ", potints_svm)
#     # print("This is points for DL: ", points_dl)
#     # print("This is total points: ", total_points)
#
#     total_svm += potints_svm
#     total_dl += points_dl
#     total += total_points
#
# print("This is points for svm: ", total_svm / total)
# print("This is points for DL: ", total_dl / total)
# print("This is total points: ", total)

"""
Single test
"""
# with open('data/categories/1525.lab') as f:
#     test_lab = f.readlines()
# test_lab = [x.strip() for x in test_lab]
#
# print("This is predict label: ", [s[0] for s in magpie.predict_from_file('data/categories/2325.txt') if s[1] >= 0.20])
# print("This is real label: ", test_lab)
