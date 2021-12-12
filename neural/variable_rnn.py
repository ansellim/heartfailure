# George Seah, Ansel Lim
# Important: prior to running this code, you should have downloaded a FastText model for Clinical Notes from the OneDrive or Google Drive link located at this Github repository: https://github.com/kexinhuang12345/clinicalBERT#gensim-word2vec-and-fasttext-models.
# Please ensure that this model is downloaded into the same directory at this code.

import re
import string
from collections import Counter

import fasttext.util
import nltk
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader

seed_everything(42, workers=True)

nltk.download('stopwords')
nltk.download('wordnet')

# Download models
fasttext.util.download_model('en', if_exists='ignore')
m1 = KeyedVectors.load("./fasttext.model")  # Downloaded from the link at github repository
m2 = fasttext.load_model('cc.en.300.bin')
fasttext.util.reduce_model(m2, 100)


# Functions for preprocessing

def get_vector(word, m1, m2):
    if word in m1.wv.key_to_index:
        # Word found in fasttext
        w_ft_2_vec = m1.wv[word]
    else:
        # Word not found in fasttext
        w_ft_2_vec = m2.get_word_vector(word)
    return w_ft_2_vec


def remove_punctuations(text):
    new_text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    new_text = new_text.replace('\n', ' ')
    return new_text


def lower_case(text):
    return text.lower()


def stopword_filter(text):
    stop = stopwords.words('english')
    return ' '.join([word for word in text.split() if word not in (stop)])


def Nchar_filter(text):
    Value = ' '.join([i for i in text.split() if len(i) >= 3])
    return Value


def remove_non_essential_words(text):
    non_essential_word_list = ['admission', 'date', 'service', 'birth', 'also']
    new_text = re.sub('[0-9]{4}pm', '', text)
    new_text = ' '.join([i for i in new_text.split() if i not in non_essential_word_list])
    return new_text


wnl = WordNetLemmatizer()


def get_pos(word):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len([item for item in w_synsets if item.pos() == "n"])
    pos_counts["v"] = len([item for item in w_synsets if item.pos() == "v"])
    pos_counts["a"] = len([item for item in w_synsets if item.pos() == "a"])
    pos_counts["r"] = len([item for item in w_synsets if item.pos() == "r"])
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]


def lemmatization(text):
    return wnl.lemmatize(text, get_pos(text))


def remove_numeric(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text


def note_to_vec(input_note):
    '''
    Input: input_notes in each row
    Return: word vector
    '''
    # Break input notes into text list
    # Convert each text into a vector
    # Append them into a matrix of n x 100, where n = number of words in input_notes
    word_list = [get_vector(i, m1, m2) for i in input_note.split()]

    return word_list


def pad_seq_array(seq_arrays, max_length):
    new_arrays = []
    for idx, seq in enumerate(seq_arrays):
        if idx % 100 == 0:
            print(idx, " note padding processed:")
        length = len(seq)
        try:
            zero_matrix = np.zeros((max_length - length, 100))
            new_seq = np.vstack((seq, zero_matrix))
            new_seq_csr_matrix = scipy.sparse.csr_matrix(new_seq)
        except:
            print("seq length is:", length, max_length)
        new_arrays.append(new_seq_csr_matrix)
    return new_arrays


# Perform preprocessing

# Set maximum sequence length for Variable RNN model

MAX_SEQUENCE_LENGTH = 2500

train = pd.read_csv("../preprocessing/processed_data/train.csv")[['text', 'label']]
val = pd.read_csv("../preprocessing/processed_data/val.csv")[['text', 'label']]
test = pd.read_csv("../preprocessing/processed_data/test.csv")[['text', 'label']]

for df in [train, val, test]:
    for function in [remove_punctuations, lower_case, stopword_filter, Nchar_filter, remove_non_essential_words,
                     lemmatization, remove_numeric]:
        df['text'] = df['text'].apply(function)

#### SOME CODE SHOULD GO HERE

train['text'] = train.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])
train['val'] = val.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])
test['val'] = test.apply(lambda x: x['text'][:MAX_SEQUENCE_LENGTH])

# Configuration for neural network training

BATCH_SIZE = 12
NUM_WORKERS = 2
NUM_EPOCHS = 5


# Create torch Datasets & DataLoaders

class HeartFailureDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        seqs_arrays = self.encodings[idx].toarray()
        labels_arrays = self.labels[idx]
        return seqs_arrays, labels_arrays

    def __len__(self):
        return len(self.labels)


train_set = HeartFailureDataset(train)
val_set = HeartFailureDataset(val)
test_set = HeartFailureDataset(test)


def to_tensors(batch):
    seqs_array = np.array([x[0] for x in batch])
    labels_array = np.array([x[1] for x in batch])
    seqs_tensor = torch.FloatTensor(seqs_array)
    labels_tensor = torch.LongTensor(labels_array)
    return seqs_tensor, labels_tensor


train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          collate_fn=to_tensors,
                          num_workers=NUM_WORKERS)
val_loader = DataLoader(dataset=val_set,
                        batch_size=BATCH_SIZE,
                        shuffle=False,
                        collate_fn=to_tensors,
                        num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_set,
                         batch_size=1,
                         shuffle=False,
                         collate_fn=to_tensors,
                         num_workers=NUM_WORKERS)

# Define neural network

criterion = nn.CrossEntropyLoss()


class MyVariableRNN(LightningModule):
    def __init__(self, dim_input):
        super(MyVariableRNN, self).__init__()
        self.FC32 = nn.Sequential(nn.Linear(dim_input, 32))
        self.GRU = nn.GRU(input_size=32, hidden_size=16,
                          num_layers=2,
                          bias=True, batch_first=True,
                          dropout=0.8,
                          bidirectional=False)
        self.FC2 = nn.Sequential(nn.Linear(16, 8),
                                 nn.ReLU(),
                                 nn.Linear(8, 2)
                                 )

    def forward(self, input_tuple):
        batch = input_tuple
        seqs = batch
        output = self.FC32(seqs)
        output = torch.tanh(output)
        h0 = torch.zeros(2,  # D* num_layers
                         output.size(0),  # batch size
                         16  # hidden size
                         , device=input_tuple.device
                         ).requires_grad_()

        output2, _ = self.GRU(output, h0.detach())
        output3 = output2[:, -1, :]
        output4 = self.FC2(output3)
        return output4

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self.forward(inputs, labels)
        loss = criterion(logits, labels)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader


# Train/val/test

checkpoint_callback = ModelCheckpoint(dirpath='./rnn',
                                      monitor='val_acc',
                                      save_top_k=-1,
                                      mode='max',
                                      filename='{epoch}-{step}-{val_acc:.2f}',
                                      auto_insert_metric_name=True,
                                      save_weights_only=True,
                                      every_n_epochs=1)

rnn = MyVariableRNN()
trainer = Trainer(max_epochs=NUM_EPOCHS,
                  check_val_every_n_epoch=1,
                  deterministic=True,
                  accelerator='auto',
                  devices='auto',
                  fast_dev_run=False,
                  callbacks=[checkpoint_callback]
                  )
trainer.fit(rnn)
trainer.test(ckpt_path="best", verbose=True)
