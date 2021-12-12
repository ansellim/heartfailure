# Ansel Lim and George Seah

# Dependencies

import logging
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from nltk import word_tokenize, sent_tokenize
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from utils import apply_basic_preprocessing

seed_everything(42, workers=True)

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.CRITICAL, ["transformers.tokenization"])

# Load dataset

print("Load dataset", datetime.now().strftime("%H:%M:%S"))
train = pd.read_csv("../preprocessing/processed_data/train.csv")[['text', 'label']]
val = pd.read_csv("../preprocessing/processed_data/val.csv")[['text', 'label']]
test = pd.read_csv("../preprocessing/processed_data/test.csv")[['text', 'label']]

# Basic preprocessing of dataset

print("Basic preprocessing of dataset", datetime.now().strftime("%H:%M:%S"))
train = apply_basic_preprocessing(train)
val = apply_basic_preprocessing(val)
test = apply_basic_preprocessing(test)

train_texts = train['text']
num_train_texts = len(train_texts)

# Additional preprocessing - remove overly common and overly uncommon words

print("Remove overly common and overly uncommon words",datetime.now().strftime("%H:%M:%S"))

tfidf = TfidfVectorizer(tokenizer=None, min_df=0.1, max_df=0.7)
tfidf.fit_transform(train_texts)


def shorten_text(text, vocabulary):
    words = word_tokenize(text, language='english')
    filtered_words = [word for word in words if word in vocabulary]
    transformed = ' '.join(filtered_words)
    return transformed


train['shortened_text'] = train.apply(lambda row: shorten_text(row['text'], tfidf.vocabulary_), axis=1)
val['shortened_text'] = val.apply(lambda row: shorten_text(row['text'], tfidf.vocabulary_), axis=1)
test['shortened_text'] = test.apply(lambda row: shorten_text(row['text'], tfidf.vocabulary_), axis=1)

# Get a base model which we will finetune; we will use a HuggingFace BERT model

print("Get base model",datetime.now().strftime("%H:%M:%S"))

model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
tokenizer = BertTokenizer.from_pretrained(model_name)
base_model = BertModel.from_pretrained(model_name, return_dict=False)

# Specify Deep Learning hyperparameters and other aspects of training configuration

SEQ_LENGTH = 512
MAX_NUM_EPOCHS = 5
BATCH_SIZE = 12
LEARNING_RATE = 5e-5
NUM_WORKERS = 6
MULTIPLIER = 10

# Modify the default tokenizer by adding tokens from the corpus

print("Add corpus-specific tokens to tokenizer",datetime.now().strftime("%H:%M:%S"))

vectorizer = TfidfVectorizer(lowercase=False, use_idf=True)
vectorizer.fit_transform(train_texts)
idf = vectorizer.idf_
indices_sorted_by_idf = sorted(range(len(idf)), key=lambda key: idf[key])
idf_sorted = idf[indices_sorted_by_idf]
tokens_by_idf = np.array(vectorizer.get_feature_names_out())[indices_sorted_by_idf]

threshold = np.percentile(idf_sorted, 70)
new_tokens = []
for token, idf in zip(tokens_by_idf, idf_sorted):
    if idf <= threshold and token.isalpha():
        new_tokens.append(token)

tokenizer.add_tokens(new_tokens)
base_model.resize_token_embeddings(len(tokenizer))

# Shuffle sentences: sentences containing key words identified by our ML models (in particular, XGBoost)
# as having high feature importance are placed nearer the front of the document

print("Shuffle sentences",datetime.now().strftime("%H:%M:%S"))

KEY_WORD_LIST = ['disp', 'aortic', 'refill', 'status', 'hospital', 'unit', 'blood', 'sig', 'stable', 'pain',
                 'valve', 'congestive', 'pressure', 'day', 'medication', 'normal', 'patient', 'release', 'drug',
                 'possible', 'history', 'rhythm',
                 'renal', 'lasix', 'sob', 'discharge', 'head', 'daily', 'admitted', 'post', 'tablet', 'heart',
                 'mellitus', 'ventricular', 'right', 'artery', 'trauma', 'hypertension', 'admission', 'needed', 'left',
                 'disease',
                 'coumadin', 'glucose', 'rehab', 'aspirin', 'trauma', 'transferred', 'nausea'
                 ]


def shuffle_sentences(text):
    '''
    @param text: a document (string) that is to be processed
    @return shuffled: a modified document (string) with the order of sentences randomly shuffled
    '''
    sentences = sent_tokenize(text)
    permuted = np.random.permutation(sentences)
    res = [any(ele in sen_token for ele in KEY_WORD_LIST) for sen_token in permuted]
    # push sentence with keyword to the front.
    # move the sentence that has no keyword to the back
    permuted_adjusted = [sen for sen, res in zip(permuted, res) if res == True]
    permuted_adjusted.extend([sen for sen, res in zip(permuted, res) if res == False])
    shuffled = ' '.join(permuted_adjusted)
    return shuffled


new_texts = []
new_labels = []
for i in range(train.shape[0]):
    label = train.loc[i, 'label']
    text = train.loc[i, 'text']
    for _ in range(MULTIPLIER):
        new_texts.append(shuffle_sentences(text))
        new_labels.append(label)

shuffled = pd.DataFrame({'label': new_labels, 'text': new_texts})
train = pd.concat([train, shuffled])
train.reset_index(inplace=True)

# Wrap the train/validation/test sets in torch Dataset classes, and then create Dataloaders to wrap around these classes

print("Create torch datasets & dataloaders",datetime.now().strftime("%H:%M:%S"))

class HeartFailureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = self.dataframe.loc[idx, 'text']
        label = self.dataframe.loc[idx, 'label']
        sample = {'text': text, 'label': label}
        return sample


train_set = HeartFailureDataset(train)
val_set = HeartFailureDataset(val)
test_set = HeartFailureDataset(test)

def preprocess(text):
    '''
    @ param text: a document (string) that is to be processed
    @return input_ids: tensor of token IDs, fed into the network
    @return attention_masks: tensor of indices which the network is to pay attention to
    '''
    encoded = tokenizer.encode_plus(text=text,
                                    add_special_tokens=True,
                                    max_length=SEQ_LENGTH,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True
                                    )
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    return input_ids, attention_masks

def collate_batch(batch):
    inputs, masks, labels = [], [], []
    for example in batch:
        input_id, attention_mask = preprocess(example['text'])
        inputs.append(input_id)
        masks.append(attention_mask)
        labels.append(example['label'])
    inputs = torch.LongTensor(inputs)
    masks = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    return (inputs, masks), labels


train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch,
                          num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
                         num_workers=NUM_WORKERS)

criterion = nn.CrossEntropyLoss()

##########################################
##########################################
######## THE ACTUAL DEEP LEARNING ########
##########################################
##########################################

############## BERT + MULTI-LAYER PERCEPTRON (FULLY CONNECTED LAYERS) ######################

# Define neural network architecture and define the dataloaders within the class definition

class BertMLP(LightningModule):
    def __init__(self, freeze_base_model=True):
        super(BertMLP, self).__init__()
        self.base = base_model
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(self.base.config.hidden_size, 256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(256, 2))
        if freeze_base_model:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, inputs, masks):
        last_hidden_state, pooled_output = self.base(input_ids=inputs, attention_mask=masks)
        logits = self.classifier(pooled_output)
        return logits

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)  # correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=500)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        (inputs, masks), labels = batch
        logits = self.forward(inputs, masks)
        loss = criterion(logits, labels)
        return loss

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader


# Train/val/test

print("Start BERT + MLP: training/validation/testing",datetime.now().strftime("%H:%M:%S"))

checkpoint_callback = ModelCheckpoint(dirpath='./mlp',
                                      monitor='val_acc',
                                      save_top_k=-1,
                                      mode='max',
                                      filename='{epoch}-{step}-{val_acc:.2f}',
                                      auto_insert_metric_name=True,
                                      save_weights_only=True,
                                      every_n_epochs=1)

mlp = BertMLP(freeze_base_model=True)

trainer = Trainer(max_epochs=MAX_NUM_EPOCHS,
                  check_val_every_n_epoch=1,
                  deterministic=True,
                  accelerator='auto',
                  devices='auto',
                  fast_dev_run=False,
                  callbacks=[checkpoint_callback])

trainer.fit(mlp)

trainer.test(ckpt_path="best", verbose=True)


################# BERT + CONVOLUTIONAL NEURAL NETWORK #########################

# Define neural network architecture and define the dataloaders within the class definition

class BertCNN(LightningModule):
    def __init__(self, freeze_base_model=True):
        super(BertCNN, self).__init__()
        self.base = base_model
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(in_features=32 * 60 * 124, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=16)
        self.fc3 = nn.Linear(16, 2)

        if freeze_base_model:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, inputs, masks):
        outputs = self.base(input_ids=inputs, attention_mask=masks)
        final_hidden_state = outputs[0].reshape(-1, 1, SEQ_LENGTH, self.base.config.hidden_size)
        out2 = self.pool(F.relu(self.conv1(final_hidden_state)))
        out2 = self.pool(F.relu(self.conv2(out2)))
        out2 = self.pool(F.relu(self.conv3(out2)))
        out2 = torch.flatten(out2, 1)
        out2 = F.dropout(out2, p=0.5)
        out2 = F.relu(self.fc1(out2))
        out2 = F.dropout(out2, p=0.2)
        out2 = F.relu(self.fc2(out2))
        logits = self.fc3(out2)
        return logits

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)  # correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=500)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        (inputs, masks), labels = batch
        logits = self.forward(inputs, masks)
        loss = criterion(logits, labels)
        return loss

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

# Train/val/test

print("Start BERT + CNN: training/validation/testing",datetime.now().strftime("%H:%M:%S"))

checkpoint_callback_cnn = ModelCheckpoint(dirpath='./cnn',
                                          monitor='val_acc',
                                          save_top_k=-1,
                                          mode='max',
                                          filename='{epoch}-{step}-{val_acc:.2f}',
                                          auto_insert_metric_name=True,
                                          save_weights_only=True,
                                          every_n_epochs=1)

cnn = BertMLP(freeze_base_model=True)

trainer_cnn = Trainer(max_epochs=MAX_NUM_EPOCHS,
                      check_val_every_n_epoch=1,
                      deterministic=True,
                      accelerator='auto',
                      devices='auto',
                      fast_dev_run=False,
                      callbacks=[checkpoint_callback_cnn])

trainer_cnn.fit(cnn)
trainer_cnn.test(ckpt_path="best", verbose=True)
