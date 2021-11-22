# Ansel Lim
# 21 Nov 2021

# The boring stuff...import dependencies, set random seeds for reproducibility, etc.

import time
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torchtext
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from torchtext.legacy.data import Field,TabularDataset,BucketIterator

# Set random seeds for reproducibility's sake.

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Torch Device will be CUDA if available, otherwise CPU.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Specify Huggingface model name (for example, "bert-base-uncased"
model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"

# Get Tokenizer from Huggingface
tokenizer = BertTokenizer.from_pretrained(model_name)

# Get model from Huggingface
bert_model = BertModel.from_pretrained(model_name)

# Set tokenizer's 'hyperparameters'
seq_length = 400 # All sequences will be padded/truncated to this sequence length

# Model specifications / hyperparameters / training settings
num_epochs = 3
num_class = 2
batch_size = 8

# Specify where to save the model later
destination_path = "./models/prototype"

# Load data
train = pd.read_csv("./datasets/train.csv")
val = pd.read_csv("./datasets/val.csv")
test = pd.read_csv("./datasets/test.csv")

### Preprocessing

# Start of "Add vocabulary" section -- Add tokens to the tokenizer's vocabulary.
# https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41

nlp = spacy.load("en_core_web_sm", exclude=['morphologizer', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

def spacy_tokenizer(document, nlp=nlp):
    # tokenize the document with spaCY
    doc = nlp(document)
    # Remove stop words and punctuation symbols
    tokens = [
        token.text for token in doc if (
        token.is_stop == False and \
        token.is_punct == False and \
        token.text.strip() != '' and \
        token.text.find("\n") == -1)]
    return tokens

tfidf_vectorizer = TfidfVectorizer(lowercase=False, tokenizer=spacy_tokenizer, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
documents = train['text']
length = len(documents)
tfidf_vectorizer.fit_transform(documents)
idf = tfidf_vectorizer.idf_
idf_sorted_indexes = sorted(range(len(idf)), key=lambda k: idf[k])
idf_sorted = idf[idf_sorted_indexes]
tokens_by_df = np.array(tfidf_vectorizer.get_feature_names())[idf_sorted_indexes]
threshold = np.percentile(idf_sorted,0.8) # ADD ONLY THE TOKENS WITH IDF IN THE BOTTOM 80TH PERCENTILE (I.E. OMIT 20% / LEAST COMMON WORDS)
new_tokens = []
for token,idf in zip(tokens_by_df,idf_sorted):
    if idf<=threshold:
        new_tokens.append(token)
print("Before adding tokens",len(tokenizer))
tokenizer.add_tokens(new_tokens)
print("After adding tokens", len(tokenizer))
bert_model.resize_token_embeddings(len(tokenizer))

vocab_size = tokenizer.vocab_size
print("Vocab size",vocab_size)

# End of "Add vocabulary" code section

# Create torch datasets from train, val, and test

class HeartFailureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self,idx):
        text = self.dataframe.loc[idx,'text']
        label = self.dataframe.loc[idx,'label']
        sample = (text,label)
        return sample

train_set = HeartFailureDataset(train)
val_set = HeartFailureDataset(val)
test_set = HeartFailureDataset(test)

# Define collate function that will be applied to each batch in the DataLoader

def collate_fn(batch):
    texts, labels = [], []
    for text,label in batch:
        texts.append(tokenizer.encode(text,max_length = seq_length,padding='max_length',truncation=True, return_tensors=None))
        labels.append(label)
    texts = torch.LongTensor(texts)
    labels = torch.LongTensor(labels)
    return texts,labels

# Create PyTorch DataLoader objects

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_set,batch_size=batch_size, shuffle=False,collate_fn = collate_fn)

# Define neural network architecture

class BerniceClassifier(nn.Module):
    def __init__(self):
        super(BerniceClassifier,self).__init__()
        self.bert = bert_model
        self.layer = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(1024,256),nn.ReLU(),nn.Dropout(p=0.5),nn.Linear(256,16),nn.ReLU(),nn.Linear(16,2))

    def forward(self,x):
        # x = torch.LongTensor(x)
        y = self.bert(x)
        pooled = y.pooler_output
        output = self.layer(pooled)

        return output

# Instantiate neural network object

bernice = BerniceClassifier()

# Define loss function, optimizer, learning rate scheduler.
learning_rate = 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bernice.parameters(),lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Functions to train the model and evaluate results (https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
def train(dataloader, model=bernice, num_epochs=5):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (text,label) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

# Evaluate function
def evaluate(dataloader,model=bernice):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, (text,label) in enumerate(dataloader):
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

# Training loop
for epoch in range(num_epochs):
    epoch_start_time = time.time()
    train(train_loader)
    accu_val = evaluate(val_loader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 50)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'validation accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 50)

# Evaluate on test set
print('Checking the results of test dataset.')
accu_test = evaluate(test_loader)
print('test accuracy {:8.3f}'.format(accu_test))
