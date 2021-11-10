# CSE6250 Project
# ANSEL LIM, 10 NOV 2021

'''
Comments:
1. Purpose of this code: Use George's CSV data, and train Huggingface pretrained models on the data to perform the binary classification task.
2. Note that prototype1.py only does the training with train and test sets; there is no validation set.

References:
1. https://www.thepythoncode.com/article/finetuning-bert-using-huggingface-transformers-python
2. https://towardsdatascience.com/fine-tuning-pretrained-nlp-models-with-huggingfaces-trainer-6326a4456e7b
'''

# Import dependencies, set random seeds for reproducibility

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import re

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

########### Specifications for model #############

'''
Specify the model name in `model_name` and the model will be downloaded from Huggingface. 
Here, I am using a BERT model pre-trained on PubMed abstracts and MIMIC-III clinical notes.
Details: https://huggingface.co/bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16
'''

model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
batch_size=4 # if model/data doesn't fit within GPU memory constraints, will need to try different batch sizes
num_labels=2
max_length =512
num_epochs = 5

# Specify where to save the model
destination_path = "./models/bluebert"

########### Utility function(s) ############

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  acc = accuracy_score(labels, preds)
  # auc = roc_auc_score(labels, preds)
  # precision = precision_score(labels, preds)
  # recall = recall_score(labels, preds)
  # f1 = f1_score(labels,preds)
  return {
      'accuracy': acc,
      # 'auc': auc,
      # 'precision': precision,
      # 'recall': recall,
      # 'f1_score': f1
  }

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

train_texts,test_texts = [tokenize(sentence) for sentence in train_texts], [tokenize(sentence) for sentence in test_texts]

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts,truncation=True,padding=True,max_length=max_length)
test_encodings = tokenizer(test_texts,truncation=True,padding=True,max_length=max_length)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self,encodings,labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MyDataset(train_encodings,train_labels)
test_dataset = MyDataset(test_encodings,test_labels)

########### Model training & evaluation ###########

model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)

model=model.to(device)

args = TrainingArguments(
    output_dir='./results',                         # output directory
    num_train_epochs=num_epochs,                    # total number of training epochs
    per_device_train_batch_size=batch_size,         # batch size per device during training
    per_device_eval_batch_size=batch_size,          # batch size for evaluation
    warmup_steps=500,                               # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                              # strength of weight decay
    logging_dir='./logs',                           # directory for storing logs
    load_best_model_at_end=True,                    # load the best model when finished training
    metric_for_best_model='accuracy',
    logging_steps=100,                              # log & save weights each logging_steps
    evaluation_strategy="steps",                    # evaluate each `logging_steps`
    eval_steps=500,
    seed=42
)

trainer = Trainer(model=model,
                  args = args,
                  train_dataset = train_dataset,
                  eval_dataset = test_dataset,
                  compute_metrics = compute_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])

trainer.train()

trainer.evaluate()

# Save model
model.save_pretrained(destination_path)
tokenizer.save_pretrained(destination_path)