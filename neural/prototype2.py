# Ansel Lim
# updated 23 Nov 2021

'''
############ NON-EXHAUSTIVE LIST OF REFERENCES #############
https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
'''

# I suggest you run the code using the conda environment specified in the environment.yml file

###################################################################################################
#######################################IMPORT DEPENDENCIES#########################################
###################################################################################################

import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from nltk import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

# Keep track of when the script started
script_start = time.time()

# Set random seeds for reproducibility's sake.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# Torch Device will be CUDA if available, otherwise CPU.
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

###################################################################################################
################CHANGE SETTINGS (except neural network architecture) HERE##########################
###################################################################################################

# If prototyping, then code runs only with a fraction of the dataset, otherwise we run the entire script with the full dataset.
prototyping = True # change to False if you want to run with full dataset

# Specify the multiplier for Preprocessing Part 3: this is the number of new documents created by shuffling each document in the train set
MULTIPLIER = 30

# Specify Huggingface model name (for example, "bert-base-uncased" or "bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16"
model_name = "bert-base-uncased"

# All sequences will be padded or truncated to SEQ_LENGTH
SEQ_LENGTH = 512

# Model specifications / hyperparameters / training settings
NUM_EPOCHS = 3 # can try 2-4
BATCH_SIZE = 8 # adjust according to memory constraints
LEARNING_RATE = 5e-5 # can try 2e-5, 3e-5, 4e-5, 5e-5

# Use a Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)

# Get a base model from Huggingface. We'll use a BERT or BERT-based model.
# We will use BertModel class rather than BertForSequenceClassification, so that we have more room for network customization for our task.
base_model = BertModel.from_pretrained(model_name)

# Specify where to save the model later
destination_path = "./models/prototype"

###################################################################################################
#########################IMPORT DATA & DEEP LEARNING-SPECIFIC PREPROCESSING########################
###################################################################################################

'''
Load data. The data has already been processed in the common preprocessing pipeline. 
In particular, note that stopword removal, lemmatization, and case conversion have already been performed as part of the common pipeline of preprocessing prior to ML/DL models.
'''
train = pd.read_csv("./datasets/train.csv")[['text', 'label']]
val = pd.read_csv("./datasets/val.csv")[['text', 'label']]
test = pd.read_csv("./datasets/test.csv")[['text', 'label']]

if prototyping:
    train = train.iloc[:30,:]
    val = val.iloc[:30,:]
    test = test.iloc[:30,:]

train_texts = train['text']
num_train_texts = len(train_texts)

'''
Preprocessing Part 1: Remove excessively common words from the train set.
I arbitrarily remove tokens which are present in more than 70% of train documents, as well as tokens that appear in less than 10% of train documents.
'''

checkpoint0 = time.time()

tfidf = TfidfVectorizer(tokenizer=None, min_df=0.1, max_df=0.7)
tfidf.fit_transform(train_texts)

def shorten_text(text, vocabulary):
    words = word_tokenize(text, language='english')
    filtered_words = [word for word in words if word in vocabulary]
    transformed = ' '.join(filtered_words)
    return transformed

train['shortened_text'] = train.apply(lambda row: shorten_text(row['text'], tfidf.vocabulary_), axis=1)

# A utility function to filter out outliers in an array using the 1.5 x IQR rule
def remove_outliers(array):
    array = np.array(array)
    q3 = np.percentile(array, 75)
    q1 = np.percentile(array, 25)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered = [val for val in array if (val >= lower) and (val <= upper)]
    return filtered

# Generate plots of lengths of train texts BEFORE and AFTER shortening by removing common words.

lengths_shortened_texts = [len(text) for text in train['shortened_text']]
lengths_shortened_texts_without_outliers = remove_outliers(lengths_shortened_texts)
lengths_original_texts = [len(text) for text in train['text']]
lengths_original_texts_without_outliers = remove_outliers(lengths_original_texts)

fig, ax = plt.subplots(2, 1)
ax = ax.flatten()
ax[0].hist(lengths_original_texts_without_outliers)
ax[0].set_title("Lengths of train texts before shortening")
ax[0].set_xlim([0, max(lengths_original_texts_without_outliers)])
ax[1].hist(lengths_shortened_texts_without_outliers)
ax[1].set_title("Lengths of train texts after shortening")
ax[1].set_xlim([0, max(lengths_original_texts_without_outliers)])
plt.tight_layout()
plt.show()

train.drop(columns={'text'},inplace=True)
train.rename(columns={'shortened_text':'text'},inplace=True)

checkpoint1 = time.time()

'''
Preprocessing Part 2: Modify tokenizer -- Add tokens to the tokenizer (fine-tune the tokenizer so that it's updated on the custom dataset). 
Reference: https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41 

Add only new tokens from train dataset which are in the bottom 70th percentile by inverse document frequency (that is, the commonest 70% of tokens). Note that this is done after preprocessing Part 1, so we are already operating on a smaller set of possible tokens.
'''

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

print("Tokenizer vocab size, before adding tokens", len(tokenizer))

tokenizer.add_tokens(new_tokens)

print("Tokenizer vocab size, after adding tokens", len(tokenizer))

# Resize the dictionary size of the embedding layer
base_model.resize_token_embeddings(len(tokenizer))

checkpoint2 = time.time()

'''
Preprocessing Part 3: Shuffle the order of sentences (data augmentation) in the train set.
'''

print("Shape of train set prior to shuffling of sentence order",train.shape)

def shuffle_sentences(text):
    '''
    @param text: a document (string) that is to be processed
    @return shuffled: a modified document (string) with the order of sentences randomly shuffled
    '''
    sentences = sent_tokenize(text)
    permuted = np.random.permutation(sentences)
    shuffled = ' '.join(permuted)
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

print("Shape of train set after shuffling of sentence order",train.shape)

checkpoint3 = time.time()

'''
Preprocessing Part 4: Create a function to tokenize a text so that the generated input_ids and attention mask may be passed into a BERT model. Then, apply this function to all texts in the train, val, and test sets.
'''
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

# For some reason, the tokenizer returns the error message "There was a bug in trie algorithm in tokenization. Attempting to recover. Please report it anyway." but it seems that the tokenization still works...And I have been unable to fix this error...But I'll plough through for now.
train[['input_ids','attention_masks']] = train.apply(lambda row: preprocess(row['text']),axis=1,result_type='expand')
val[['input_ids','attention_masks']] = val.apply(lambda row: preprocess(row['text']),axis=1,result_type='expand')
test[['input_ids','attention_masks']] = test.apply(lambda row: preprocess(row['text']),axis=1,result_type='expand')

checkpoint4 = time.time()

print("Preprocessing part 4 took {} seconds".format(checkpoint4-checkpoint3))

'''
Preprocessing Part 5:
- Create torch Datasets from train, val, and test
- Create torch DataLoaders for train, val, and test, which wrap around the Datasets
'''

# Design and create torch Datasets

class HeartFailureDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        text = self.dataframe.loc[idx, 'text']
        label = self.dataframe.loc[idx, 'label']
        input_ids = self.dataframe.loc[idx,'input_ids']
        attention_masks = self.dataframe.loc[idx,'attention_masks']
        sample = {'text': text, 'input_ids': input_ids, 'attention_masks': attention_masks, 'label': label}
        return sample


train_set = HeartFailureDataset(train)
val_set = HeartFailureDataset(val)
test_set = HeartFailureDataset(test)

# Define collate function that will be applied to each batch in the DataLoader

def collate_batch(batch):
    inputs, masks, labels = [], [], []
    for example in batch:
        inputs.append(example['input_ids'])
        masks.append(example['attention_masks'])
        labels.append(example['label'])
    inputs = torch.LongTensor(inputs)
    masks = torch.LongTensor(masks)
    labels = torch.LongTensor(labels)
    return (inputs,masks),labels

# Create PyTorch DataLoader objects

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

checkpoint5 = time.time()

###################################################################################################
#########################DESIGN & DEFINE NEURAL NETWORK ARCHITECTURE HERE##########################
###################################################################################################

# Define neural network architecture here.

class Network(nn.Module):
    def __init__(self,freeze_base_model=True):
        super(Network, self).__init__()
        self.base = base_model
        self.clf = nn.Sequential(nn.Dropout(p=0.5),
                                 nn.Linear(768, 256), nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(256, 16), nn.ReLU(),
                                 nn.Linear(16, 2))
        if freeze_base_model:
            for param in self.base.parameters():
                param.requires_grad = False
    def forward(self, inputs, masks):
        outputs = self.base(input_ids = inputs, attention_mask = masks)
        final_hidden_state = outputs[0][:,0,:]
        logits = self.clf(final_hidden_state)
        return logits

###################################################################################################
##############################TRAIN AND EVALUATE MODEL#############################################
###################################################################################################

checkpoint6 = time.time()

# Instantiate neural network object

model = Network(freeze_base_model=True)
model = model.to(device)

# Define loss function, optimizer, learning rate scheduler.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

# Functions to train the model and evaluate results (copied and adapted from PyTorch documentation)
def train(dataloader, model, log_interval):
    '''
    @param dataloader: train_loader, val_loader, or test_loader
    @param log_interval: track the performance every log_interval batches.
    '''
    model.train()
    total_acc, total_count = 0, 0
    start_time = time.time()
    for idx, ((inputs,masks),labels) in enumerate(dataloader):
        inputs,masks,labels = inputs.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()
        predicted_labels = model(inputs,masks)
        loss = criterion(predicted_labels, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_labels.argmax(1) == labels).sum().item()
        total_count += labels.size(0)
        if idx % log_interval == 0 and idx > 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc / total_count))
            total_acc, total_count = 0, 0
    elapsed_time = time.time()
    print("Completed training in {} seconds".format(elapsed_time - start_time))

# Evaluate function - Currently this is only for ACCURACY, but we can change it to focus on F1 / AUC.
def evaluate(dataloader, model):
    model.eval()
    total_acc, total_count = 0, 0
    with torch.no_grad():
        for idx, ((inputs,masks),labels) in enumerate(dataloader):
            inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
            predicted_labels = model(inputs,masks)
            total_acc += (predicted_labels.argmax(1) == labels).sum().item()
            total_count += labels.size(0)
    return total_acc / total_count

# Training loop
for epoch in range(NUM_EPOCHS):
    total_accu = None
    epoch_start_time = time.time()
    train(train_loader, model=model, log_interval=1)
    accu_val = evaluate(val_loader, model = model)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 50)
    print('| end of epoch {:3d} | time: {:5.2f}s | validation accuracy {:8.3f} '.format(epoch,time.time() - epoch_start_time,accu_val))
    print('-' * 50)

# Evaluate on test set
print('Checking the results of test dataset.')
test_accuracy = evaluate(test_loader,model)
print('test accuracy {:8.3f}'.format(test_accuracy))

final_checkpoint = time.time()
print("Training and testing completed in {} seconds".format(final_checkpoint - checkpoint6))
print("Script completed in {} seconds".format(final_checkpoint - script_start))