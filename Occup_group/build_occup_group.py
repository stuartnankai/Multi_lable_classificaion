import logging
import os
import pandas as pd
from bs4 import BeautifulSoup
from langdetect import *
from langdetect import lang_detect_exception
from translate import Translator
from nltk import word_tokenize

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/sunxuan/Desktop/streamtest-637200dc2640.json'


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


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    # translate_client = translate.Client()
    #
    # if isinstance(text, six.binary_type):
    #     text = text.decode('utf-8')
    #
    # # Text can also be a sequence of strings, in which case this method
    # # will return a sequence of results for each text.
    # result = translate_client.translate(
    #     text, target_language=target)
    #
    # print(u'Text: {}'.format(result['input']))
    # print(u'Translation: {}'.format(result['translatedText']))
    # print(u'Detected source language: {}'.format(
    #     result['detectedSourceLanguage']))
    # return result['translatedText']

    # Try translator package
    translator = Translator(to_lang=target)
    result = translator.translate(text)
    print("This is translate result : ", result)
    return result


def detect_lan(text):
    lang = detect(text)
    if lang == 'en':
        return text
    else:
        print("This is not english!!!!!!")
        return translate_text('en', text)
        # return "NOT ENGLISH"


def detect_language(text):
    # if not text or len(text) < 2:
    #     return None
    try:
        lang = detect(text)
        # print("This is detected lang: ", lang )
        if lang == 'en':
            return text
        else:
            print("This is not english!!!!!!", text)
            return translate_text('en', text)
    except lang_detect_exception.LangDetectException as e:
        return None


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


def order(frame, var):
    varlist = [w for w in frame.columns if w not in var]
    frame = frame[var + varlist]
    return frame


def build_label(file):
    label_list = file.tolist()
    label_set = set(label_list)
    temp_dict = {}
    j = 1
    for i in label_set:
        temp_dict[i] = j
        j += 1
    print("This is temp_dict: ", temp_dict)
    return temp_dict


def clean_job(df):
    df_english = pd.DataFrame(data=None, columns=df.columns)
    for i in range(len(df)):
        text_descrip = df.at[i, 'description']
        text_title = df.loc[i, 'title']
        occupation_group = df.loc[i, 'untouched_occupational_group']
        if is_english(text_title) and not isNaN(text_descrip) and not isNaN(occupation_group):
            # print("Ready to add new row : ", df.loc[i])
            df.loc[i, 'description'] = convert_description(text_descrip)
            df.loc[i,'title'] = word_tokenize(text_title)
            df_english = df_english.append(df.loc[i])
    # print("This is document length: ", len(df_english))  # No duplicates
    # df_english[['title', 'description', 'grade_id', 'origin_id', 'source_id', 'untouched_occupational_group']].to_csv(
    #     'cleaned_10000.csv', sep=',', encoding='utf-8')
    return df_english[['title', 'description', 'untouched_occupational_group']]


def build_new_data(df, dict):
    df['label'] = df.apply(lambda row: dict[row.untouched_occupational_group],
                           axis=1)  # get label for each job
    print("This is : ", df['label'].tolist())
    return df


df = pd.read_csv('/Users/sunxuan/Documents/Impactpool/Untitled_20180201.csv')
df_english = clean_job(df)
df_english.to_csv('Occup_group/cleaned_10000.csv',
                  sep=',', encoding='utf-8',
                  index=False)

temp_dict = build_label(df_english['untouched_occupational_group'])
temp_df = build_new_data(df_english, temp_dict)
temp_df.to_csv('Occup_group/cleaned_label_10000.csv', sep=',', encoding='utf-8', index=False)
print("This is : ", temp_df[['title','label']])

