# Ansel Lim
# Python code for creating datasets from the data extracted via BigQuery (bq_data_in_csv)

import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(43)
np.random.seed(43)

########### Additional preprocessing, etc. ###########

positives = pd.read_csv("../bigquery/bq_data_in_csv/hf_positives.csv")
positives['label'] = 1
negatives = pd.read_csv("../bigquery/bq_data_in_csv/hf_negative.csv")
negatives['label'] = 0
all_patients = pd.concat([positives, negatives], axis=0)

labels = all_patients.groupby('SUBJECT_ID').agg({'label': max}).reset_index()
texts = all_patients.groupby('SUBJECT_ID')['clean_text'].apply(lambda x: ''.join(x)).reset_index()
texts_and_labels = texts.set_index('SUBJECT_ID').join(labels.set_index('SUBJECT_ID')).reset_index()

train_df, test_and_val_df = train_test_split(texts_and_labels, test_size=0.4, random_state=42)
test_df, val_df = train_test_split(test_and_val_df, test_size=0.5, random_state=42)

train_texts, val_texts, test_texts = train_df['clean_text'].tolist(), val_df['clean_text'].tolist(), test_df[
    'clean_text'].tolist()

train_labels, val_labels, test_labels = train_df['label'].tolist(), val_df['label'].tolist(), test_df['label'].tolist()

data = dict()
data['train_texts'], data['val_texts'], data['test_texts'] = train_texts, val_texts, test_texts
data['train_labels'], data['val_labels'], data['test_labels'] = train_labels, val_labels, test_labels

train = pd.DataFrame({'text': data['train_texts'], 'label': data['train_labels']})
val = pd.DataFrame({'text':data['val_texts'],'label':data['val_labels']})
test = pd.DataFrame({'text':data['test_texts'],'label':data['test_labels']})

train.to_csv("./datasets/train.csv")
val.to_csv("./datasets/val.csv")
test.to_csv("./datasets/test.csv")