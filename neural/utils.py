import re
from collections import Counter

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')


def lower_case(text):
    '''
    Convert text to lowercase
    '''
    return text.lower()


def clinical_text_preprocessing(text):
    '''
    Remove unnecessary/uninfomrative words, headers
    '''
    text = re.sub('admission date:', '', text)
    text = re.sub('discharge date:', '', text)
    text = re.sub('date of birth:', '', text)
    text = re.sub('service:', '', text)
    text = re.sub('sex:', '', text)
    text = re.sub('admission', '', text)
    text = re.sub('allergies:', '', text)
    text = re.sub('attending:', '', text)
    text = re.sub('chief complaint:', '', text)
    text = re.sub('major surgical or invasive procedure:', 'procedure', text)
    text = re.sub('\?\?\?\?\?\?', '', text)
    text = re.sub('discharge medications:', 'medications', text)
    text = re.sub('discharge disposition:', 'disposition', text)
    text = re.sub('discharge', '', text)
    text = re.sub('completed by:', '', text)
    text = re.sub('\\[(.*?)\\]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub('--|__|==', '', text)
    text = re.sub('also', '', text)
    return text


def remove_short_words(text):
    '''
    Remove overly short words which may not have much meaning
    '''
    text = ' '.join([i for i in text.split() if len(i) >= 3])
    return text


wnl = WordNetLemmatizer()


def get_pos(word):
    '''
    For each word, use WordNet to get the most common synonym (which may be itself)
    '''
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def lemmatization(text):
    '''
    Lemmatize the text using the most common synonyms of words
    '''
    return wnl.lemmatize(text, get_pos(text))


def remove_numeric(text):
    text = re.sub('[0-9]', ' ', text)
    return text


def remove_punctuations(text):
    # remove punctuation and enter
    new_text = re.sub('[!"\\#\\$%\\&\'\\(\\)\\*\\+,\\-/:;<=>\\?@\\[\\\\\\]\\^_`\\{\\|\\}\\~]', '', text)
    new_text = new_text.replace('\n', ' ')
    return new_text


def apply_basic_preprocessing(data_df):
    '''
    Apply the preprocessing steps
    '''
    _int_data_df = data_df.copy()
    _int_data_df['text'] = _int_data_df['text'].apply(lower_case)
    _int_data_df['text'] = _int_data_df['text'].apply(clinical_text_preprocessing)
    _int_data_df['text'] = _int_data_df['text'].apply(remove_short_words)
    _int_data_df['text'] = _int_data_df['text'].apply(lemmatization)
    _int_data_df['text'] = _int_data_df['text'].apply(remove_numeric)
    _int_data_df['text'] = _int_data_df['text'].apply(remove_punctuations)
    return _int_data_df
