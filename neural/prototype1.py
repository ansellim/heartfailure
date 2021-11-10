# ANSEL LIM, 10 NOV 2021
# With heavy reference to online tutorial: https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics import roc_auc_score
from datasets import Dataset
import re
import pickle

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

########### Additional preprocessing, etc. ###########

positives = pd.read_csv("../bigquery/bq_data_in_csv/hf_positives.csv")
positives['label']=1
negatives = pd.read_csv("../bigquery/bq_data_in_csv/hf_negative.csv")
negatives['label']=0
all_patients = pd.concat([positives,negatives],axis=0)

labels = all_patients.groupby('SUBJECT_ID').agg({'label':max}).reset_index()
texts = all_patients.groupby('SUBJECT_ID')['clean_text'].apply(lambda x:''.join(x)).reset_index()
texts_and_labels = texts.set_index('SUBJECT_ID').join(labels.set_index('SUBJECT_ID')).reset_index()

train_df,test_df = train_test_split(texts_and_labels,test_size=0.3,random_state=42)

(train_texts,test_texts,train_labels,test_labels) = train_df['clean_text'].tolist(), test_df['clean_text'].tolist(), train_df['label'].tolist(), test_df['label'].tolist()

def tokenize(string):
    string = string.lower()
    string = re.sub(r'[\r\n]+', ' ', string)
    string = re.sub(r'[^\x00-\x7F]+', ' ', string)
    tokenized = TreebankWordTokenizer().tokenize(string)
    sentence = ' '.join(tokenized)
    sentence = re.sub(r"\s's\b", "'s", sentence)
    return sentence

train_texts,test_texts = [tokenize(sentence) for sentence in train_texts], [tokenize(sentence) for sentence in test_texts]

########### Model training & evaluation ###########

# model_name = "bert-base-uncased"
model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
batch_size=4 # reduced batch size from 8 to 4 to try to fit GPU memory constraints
num_labels=2
max_length =512

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)

model=model.to(device)

train_encodings = tokenizer(train_texts,truncation=True,padding=True,max_length=max_length)
test_encodings = tokenizer(test_texts,truncation=True,padding=True,max_length=max_length)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings,train_labels)
test_dataset = MyDataset(test_encodings,test_labels)

# train_dataset = train_dataset.to(device)
# test_dataset = test_dataset.to(device)

training_args = TrainingArguments(
    output_dir='./results',                         # output directory
    num_train_epochs=5,                             # total number of training epochs
    per_device_train_batch_size=batch_size,          # batch size per device during training
    per_device_eval_batch_size=batch_size,          # batch size for evaluation
    warmup_steps=500,                                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                               # strength of weight decay
    logging_dir='./logs',                           # directory for storing logs
    load_best_model_at_end=True,                    # load the best model when finished training
    metric_for_best_model='accuracy',
    logging_steps=100,                              # log & save weights each logging_steps
    evaluation_strategy="steps",                    # evaluate each `logging_steps`
)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

trainer = Trainer(model=model,
                  args = training_args,
                  train_dataset = train_dataset,
                  eval_dataset = test_dataset,
                  compute_metrics = compute_metrics)

trainer.train()

trainer.evaluate()

model_path = "./models/bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)