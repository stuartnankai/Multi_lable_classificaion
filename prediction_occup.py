import pandas as pd
from random import randint
from bs4 import BeautifulSoup
from langdetect import *
from langdetect import lang_detect_exception
from nltk.corpus import stopwords
from nltk import word_tokenize, re, RegexpTokenizer
import ast
import os

# Load NLTK's English stop-words list
stop_words = set(stopwords.words('english'))

from magpie import Magpie


# html to text
def convert_description(text):
    soup = BeautifulSoup(text, 'lxml')
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)
    # print("This is text before translation: ", text )
    # text = detect_lan(text)
    return text


def is_english(text):
    try:
        lang = detect(text)
        if lang == 'en':
            return True
        else:
            return False
    except lang_detect_exception.LangDetectException as e:
        return False


def isNaN(num):
    return num != num


def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return " ".join(filtered_words)


def clean_job(df, df_new):
    df_english = pd.DataFrame(data=None)
    for i in range(len(df_new)):
        text_descrip = df.loc[df['id'] == df_new.index[i], 'description'].values[0]
        text_title = df.loc[df['id'] == df_new.index[i], 'title'].values[0]
        if is_english(text_title) and not isNaN(text_descrip):
            df_english = df_english.append({'id': df_new.index[i], 'title': preprocess(text_title),
                                            'description': preprocess(convert_description(text_descrip)),
                                            'group_id': df_new[df_new.index[i]]}, ignore_index=True)
    return df_english[['id', 'title', 'description', 'group_id']]


def get_score(results_dl, real_result):
    points_dl = 0
    total_points = 0

    for title, label in real_result.items():
        # print("This is svm: ",  )
        bonus_dl = len([i for i in label if i in results_dl.get(title)])

        punish_dl = len([i for i in label if i not in results_dl.get(title)])

        # print("This is punish svm: ", punish_svm)
        # print("This is punish dl: ", punish_dl)

        # potints_svm -= punish_svm
        points_dl += bonus_dl
        points_dl -= punish_dl
        total_points += len(label)
        # print("This is svm: ", [i for i in label if i in results_svm.get(title)])
        # print("This is dl: ", [i for i in label if i in results_dl.get(title)])
    return points_dl,total_points


"""
Prepare corpus of labeled data 
"""

def build_dataset(df1):
# Build the dataset
    df_new = df1.copy()
    df_new = df_new.groupby('id')['occupational_group_id'].apply(list)

    df_english = clean_job(df1, df_new)

    print("This is : ", df_english.head())

    df_english.to_csv('data/cleaned_18000.csv',
                      sep=',', encoding='utf-8',
                      index=False)
    #
    df = pd.read_csv('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/Occup_group/cleaned_18000.csv', sep=',')
    df = df.drop(['id'], axis=1)
    lab_list = []
    for i, row in df.iterrows():
        if i > len(df):
            break
        else:
            file_name = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories/' + str(i) + '.txt'
            lab_name = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories/' + str(i) + '.lab'

            title_data = df.at[i, 'title'].encode('ascii', 'ignore').decode('ascii')

            with open(file_name, 'w') as the_file:
                the_file.write(title_data)

            row_data = eval(df.at[i, 'group_id'])
            for j in row_data:
                lab_list.append(j)
                with open(lab_name, 'a') as the_file:
                    the_file.write(str(j) + '\n')
    lab_set = list(set(lab_list))
    file = '/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/' + 'categories' + '.labels'
    for i in lab_set:
        with open(file, 'a') as the_file:
            the_file.write(str(i) + '\n')

def train_dl(save,vec_dim,epochs):
    """
    train process
    """
    magpie = Magpie()

    # magpie.train_word2vec('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', vec_dim=100)
    # magpie.fit_scaler('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories')
    magpie.init_word_vectors('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', vec_dim=vec_dim)

    with open('data/categories.labels') as f:
        labels = f.readlines()
    labels = [x.strip() for x in labels]
    magpie.train('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories', labels, test_ratio=0.0, epochs=epochs)

    if save:
        """
        Save model
        """
        magpie.save_word2vec_model('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/embeddings/here')
        magpie.save_scaler('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/scaler/here', overwrite=True)
        magpie.save_model('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/model/here.h5')
    return magpie

def reinitialize():
    """
    Reinitialize
    """
    with open('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/categories.labels') as f:  # job labels
        labels = f.readlines()
    labels = [x.strip() for x in labels]

    magpie = Magpie(
        keras_model='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/model/here.h5',
        word2vec_model='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/embeddings/here',
        scaler='/Users/sunxuan/Documents/PycharmProjects/ImpactPool/data/save/scaler/here',
        labels=labels
    )
    return magpie

def test_accuracy(x,magpie,df):
    results_dl = {}
    real_result = {}
    for i in x:
        file_id = i
        file_path = 'data/categories/' + str(file_id) + '.lab'
        title_path = 'data/categories/' + str(file_id) + '.txt'
        with open(file_path) as f:
            test_lab = f.readlines()
        test_lab = [x.strip() for x in test_lab]

        with open(title_path) as tf:
            title_name = tf.readlines()
        # for text, using magpie.predict_from_text()
        # for file, using magpie.predict_from_file()
        predict_list = [s[0] for s in magpie.predict_from_text(title_name[0]) if s[1] >= 0.20]
        print("This is predict label: ", predict_list)
        print("This is real label: ", test_lab)
        print("This is predict occupational group name: ",
              [df.loc[df['id'] == int(i), 'name'].iloc[0] for i in predict_list])
        print("This is real occupational group name: ",
              [df.loc[df['id'] == int(i), 'name'].iloc[0] for i in test_lab])
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        real_result[title_name[0]] = test_lab
        results_dl[title_name[0]] = predict_list

    points_dl, total_points = get_score(results_dl, real_result)
    print("This is points for DL: ", points_dl)
    print("This is total points: ", total_points)
    print("This is accuracy: ", points_dl / total_points)

"""
Prediction
"""


if __name__ == '__main__':
    use_model = True
    test_size = 200
    df = pd.read_csv('/Users/sunxuan/Documents/PycharmProjects/ImpactPool/Untitled_20180215.csv', sep=',')
    DIR = 'data/categories'
    file_size = 0.5 * (len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])) - 1
    x = [randint(0, file_size) for p in
         range(0, test_size)]
    if use_model:
        magpie = reinitialize()
        test_accuracy(x,magpie,df)
    else:
        # df1 = pd.read_csv('get_job_info_20180215.csv')
        magpie = train_dl(save=False,vec_dim=100,epochs=5)
        test_accuracy(x,magpie,df)
        # title_name = input("Please type the job title:")
        # print("This is predict label: ", [s[0] for s in magpie.predict_from_text(title_name) if s[1] >= 0.20])
