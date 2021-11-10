# ANSEL LIM, 10 NOV 2021
# Uses George's "bq_data_in_csv" and splits the data into train/val/test sets.

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle
import re
from nltk.tokenize import TreebankWordTokenizer

random.seed(42)
np.random.seed(42)

########### Additional preprocessing, etc. ###########

positives = pd.read_csv("../bigquery/bq_data_in_csv/hf_positives.csv")
positives['label']=1
negatives = pd.read_csv("../bigquery/bq_data_in_csv/hf_negative.csv")
negatives['label']=0
all_patients = pd.concat([positives,negatives],axis=0)

labels = all_patients.groupby('SUBJECT_ID').agg({'label':max}).reset_index()
texts = all_patients.groupby('SUBJECT_ID')['clean_text'].apply(lambda x:''.join(x)).reset_index()
texts_and_labels = texts.set_index('SUBJECT_ID').join(labels.set_index('SUBJECT_ID')).reset_index()

train_df,test_val_df = train_test_split(texts_and_labels, test_size=0.4, random_state=42)
test_df, val_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

train_texts,val_texts,test_texts = train_df['clean_text'].tolist(), val_df['clean_text'].tolist(), test_df['clean_text'].tolist()

train_labels,val_labels,test_labels = train_df['label'].tolist(), val_df['label'].tolist(), test_df['label'].tolist()

'''
The tokenize() method reuses the code in the BlueBERT PubMed MIMIC documentation (https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16).
Steps involved in tokenize():
1. Lowercasing the text
2. Removing special characters
3. Tokenizing the text using NLTK Treebank tokenizer
'''

def tokenize(string):
    string = string.lower()
    string = re.sub(r'[\r\n]+', ' ', string)
    string = re.sub(r'[^\x00-\x7F]+', ' ', string)
    tokenized = TreebankWordTokenizer().tokenize(string)
    sentence = ' '.join(tokenized)
    sentence = re.sub(r"\s's\b", "'s", sentence)
    return sentence

train_texts, val_texts, test_texts = [tokenize(sentence) for sentence in train_texts], [tokenize(sentence) for sentence in val_texts], [tokenize(sentence) for sentence in test_texts]

data = dict()
data['train_texts'], data['val_texts'], data['test_texts'] = train_texts,val_texts,test_texts
data['train_labels'], data['val_labels'], data['test_labels'] = train_labels,val_labels,test_labels

with open("./datasets/data.pkl", "wb") as file:
    pickle.dump(data, file)