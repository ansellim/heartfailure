import requests
import os

import transformers
from dotenv import load_dotenv
import pandas as pd
import torch
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet
from collections import Counter
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from requests.structures import CaseInsensitiveDict
import torch.nn as nn
import logging
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, LightningDataModule

load_dotenv()
PROJECT_ID = os.getenv('PROJECT_ID')
TOKEN = os.getenv('TOKEN')

headers = CaseInsensitiveDict()
headers['Authorization'] = "Bearer {}".format(TOKEN)

train = pd.read_csv("./datasets/train.csv")[['text', 'label']]
val = pd.read_csv("./datasets/val.csv")[['text', 'label']]
test = pd.read_csv("./datasets/test.csv")[['text', 'label']]

def lower_case(text):
        return text.lower()

def clinical_text_preprocessing(text):
    text = re.sub('admission date:','',text)
    text = re.sub('discharge date:', '', text)
    text = re.sub('date of birth:','',text)
    text = re.sub('service:','',text)
    text = re.sub('sex:', '', text)
    text = re.sub('admission','',text)
    text = re.sub('allergies:','',text)
    text = re.sub('attending:','',text)
    text = re.sub('chief complaint:','',text)
    text = re.sub('major surgical or invasive procedure:','procedure',text)
    text = re.sub('\?\?\?\?\?\?', '', text)
    text = re.sub('discharge medications:', 'medications',text)
    text = re.sub('discharge disposition:','disposition',text)
    text = re.sub('discharge','',text)
    text = re.sub('completed by:','',text)
    text = re.sub('\\[(.*?)\\]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub('--|__|==', '', text)
    text = re.sub('also','',text)
    return text

def remove_short_words(text):
    text = ' '.join([i for i in text.split() if len(i) >= 3])
    return text

wnl = WordNetLemmatizer()

def get_pos( word ):
    w_synsets = wordnet.synsets(word)
    pos_counts = Counter()
    pos_counts["n"] = len(  [ item for item in w_synsets if item.pos()=="n"]  )
    pos_counts["v"] = len(  [ item for item in w_synsets if item.pos()=="v"]  )
    pos_counts["a"] = len(  [ item for item in w_synsets if item.pos()=="a"]  )
    pos_counts["r"] = len(  [ item for item in w_synsets if item.pos()=="r"]  )
    most_common_pos_list = pos_counts.most_common(3)
    return most_common_pos_list[0][0]

def lemmatization(text):
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
        _int_data_df = data_df.copy()
        _int_data_df['text'] = _int_data_df['text'].apply(lower_case)
        _int_data_df['text'] = _int_data_df['text'].apply(clinical_text_preprocessing)
        _int_data_df['text'] = _int_data_df['text'].apply(remove_short_words)
        _int_data_df['text'] = _int_data_df['text'].apply(lemmatization)
        _int_data_df['text'] = _int_data_df['text'].apply(remove_numeric)
        _int_data_df['text'] = _int_data_df['text'].apply(remove_punctuations)
        return _int_data_df

train = apply_basic_preprocessing(train)
val = apply_basic_preprocessing(val)
test = apply_basic_preprocessing(test)

train_texts = train['text']
num_train_texts = len(train_texts)

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

model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
SEQ_LENGTH = 512
NUM_EPOCHS = 3
BATCH_SIZE = 12
LEARNING_RATE = 5e-5
NUM_WORKERS = 6
MULTIPLIER = 10
tokenizer = BertTokenizer.from_pretrained(model_name)
base_model = BertModel.from_pretrained(model_name,return_dict = False)

tokenizer.add_tokens(new_tokens)
base_model.resize_token_embeddings(len(tokenizer))

KEY_WORD_LIST = ['disp','aortic','refill','status','hospital','unit','blood','sig','stable','pain',
 'valve','congestive','pressure','day','medication','normal','patient','release','drug','possible','history','rhythm',
 'renal','lasix','sob','discharge','head','daily','admitted','post','tablet','heart',
 'mellitus','ventricular','right','artery','trauma','hypertension','admission','needed','left','disease',
 'coumadin','glucose','rehab','aspirin','trauma','transferred','nausea'
]

def shuffle_sentences(text):
    '''
    @param text: a document (string) that is to be processed
    @return shuffled: a modified document (string) with the order of sentences randomly shuffled
    '''
    sentences = sent_tokenize(text)
    permuted = np.random.permutation(sentences)
    res = [any(ele in sen_token for ele in KEY_WORD_LIST) for sen_token in permuted]
    #push sentence with keyword to the front.
    #move the sentence that has no keyword to the back
    permuted_adjusted = [sen for sen,res in zip(permuted,res) if res==True]
    permuted_adjusted.extend([sen for sen,res in zip(permuted,res) if res==False])
    shuffled = ' '.join(permuted_adjusted)
    return shuffled

new_texts=[]
new_labels=[]
for i in range(train.shape[0]):
    label = train.loc[i,'label']
    text = train.loc[i,'text']
    for _ in range(MULTIPLIER):
        new_texts.append(shuffle_sentences(text))
        new_labels.append(label)

shuffled = pd.DataFrame({'label':new_labels,'text':new_texts})
train = pd.concat([train,shuffled])
train.reset_index(inplace=True)

def preprocess(text):
    '''
    @ param text: a document (string) that is to be processed
    @return input_ids: tensor of token IDs, fed into the network
    @return attention_masks: tensor of indices which the network is to pay attention to
    '''
    encoded = tokenizer.encode_plus(text = text,
                                    add_special_tokens=True,
                                    max_length=SEQ_LENGTH,
                                    padding='max_length',
                                    truncation=True,
                                    return_attention_mask=True
                                    )
    input_ids = encoded['input_ids']
    attention_masks = encoded['attention_mask']
    return input_ids,attention_masks

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

def collate_batch(batch):
    inputs, masks, labels = [], [], []
    for example in batch:
        input_id,attention_mask = preprocess(example['text'])
        inputs.append(input_id)
        masks.append(attention_mask)
        labels.append(example['label'])
    inputs = torch.LongTensor(inputs)
    masks = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    return (inputs,masks),labels

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch,num_workers=NUM_WORKERS)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch,num_workers=NUM_WORKERS)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch,num_workers=NUM_WORKERS)

criterion = nn.CrossEntropyLoss()

class BertMLP(LightningModule):
    def __init__(self,freeze_base_model=True):
        super(BertMLP,self).__init__()
        self.base = base_model
        self.classifier = nn.Sequential(nn.Dropout(p=0.5),
                                        nn.Linear(self.base.config.hidden_size,256),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.3),
                                        nn.Linear(256,2))
        if freeze_base_model:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self,inputs,masks):
        last_hidden_state,pooled_output = self.base(input_ids = inputs, attention_mask = masks)
        logits = self.classifier(pooled_output)
        return logits

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),lr=2e-5)#correct_bias=False)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=500)

        return [optimizer],[scheduler]

    def training_step(self,batch,batch_idx):
        (inputs,masks),labels = batch
        print(inputs.size())
        print(labels.size())
        logits = self.forward(inputs,masks)
        loss = criterion(logits,labels)

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return val_loader

    def test_dataloader(self):
        return test_loader

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
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.CRITICAL, ["transformers.tokenization"])

model = BertMLP(freeze_base_model=True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device used",device)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)

trainer=Trainer()
trainer.fit(model,max_epochs=10)
trainer.test(ckpt_path="best")