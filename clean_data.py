import urllib
import six
from langdetect import detect
from google.cloud import translate
import pandas as pd
from translate import Translator
import lxml.html.clean
import numpy as np
import nltk
from bs4 import BeautifulSoup
import os
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/sunxuan/Desktop/streamtest-637200dc2640.json'

df = pd.read_csv('jobtemp.csv', delimiter=',', header=None)
df.columns = ["title", "description", "id"]


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
    text = detect_lan(text)
    return text


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(
        text, target_language=target)

    print(u'Text: {}'.format(result['input']))
    print(u'Translation: {}'.format(result['translatedText']))
    print(u'Detected source language: {}'.format(
        result['detectedSourceLanguage']))
    return result['translatedText']

    # Try translator package
    # translator = Translator(to_lang=target)
    # result = translator.translate(text)

    # print("This is result: ", result)
    # return result


def detect_lan(text):
    lang = detect(text)
    if lang == 'en':
        return text
    else:
        print("This is not english!!!!!!")
        return translate_text("en", text)
        # return "NOT ENGLISH"

def isNaN(num):
    return num != num

def order(frame,var):
    varlist =[w for w in frame.columns if w not in var]
    frame = frame[var+varlist]
    return frame

for i in range(len(df.index)):
    text_descrip = df.at[i,'description']
    text_title = df.at[i, 'title']
    if isNaN(text_descrip):
        df.at[i, 'text'] = text_descrip
    else:
        df.at[i, 'text'] = convert_description(text_descrip)
    if isNaN(text_title):
        df.at[i,'en_title'] = text_title
    else:
        df.at[i, 'en_title'] = detect_lan(text_title)

df = order(df,['text'])
df = order(df,['en_title'])
# print("This is : ", df)
df.to_csv('cleantext.csv', sep='\t', encoding='utf-8')