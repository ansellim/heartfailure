# adapted from George's codes
from gensim.models import KeyedVectors
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
import pickle
import re, string
import scipy

#please download this beforehand
#this will take 30 min++ to download
# fasttext.util.download_model('en', if_exists='ignore')  # English

# #please download this beforehand
# #Update this portion to your local
# m1 = KeyedVectors.load(r'C:\Users\jolee\Desktop\OMSA\CSE6250\project\CSE6250_Project\word2vec+fastText\FastText\fasttext.model')

# #loading the model can take some time
# m2 = fasttext.load_model(r'C:\Users\jolee\Desktop\OMSA\CSE6250\project\CSE6250_Project\cc.en.300.bin')
# fasttext.util.reduce_model(m2, 100)

def get_vector(word,m1,m2):
    if word in m1.wv.key_to_index:
        w_ft_2_vec = m1.wv[word]

    else:
        #print("word not found in clinical note FastText")
        w_ft_2_vec = m2.get_word_vector(word)
    return w_ft_2_vec

### next step:
### ###
## Add the method to convert clinical note to word embedding array
### ###

def note_to_vec(input_note):
    """
    Input: input_notes in each row
    Return: word vector
    """
    #1 break input notes into text list
    #2 convert each text into a vector
    #Append them into a matrix of n x 100 ,where n = number of words in input_notes
    word_list = [get_vector(i,m1,m2) for i in input_note.split()]
    return np.array(word_list)

def create_dataset(text):
    #convert each note to word embedding
    seq_arrays = []
    for idx,note in enumerate(text):
        if idx % 100==0:
            print(idx,"notes embedding processed")
        seq_arrays.append(note_to_vec(note))

    return seq_arrays

def generate_w2v_features(text):
    data = create_dataset(text)

    ## average features for each patient
    ## https://zablo.net/blog/post/twitter-sentiment-analysis-python-scikit-word2vec-nltk-xgboost/
    new_data = []
    for s in data:
        new_data.append(s.mean(axis = 0))

    new_data = np.vstack(new_data)
    
    return new_data

def generate_w2v_features():
    prefix = "../pre-processing/dataset_lemma_avg_v3/"

    w2v_features = []
    for l in ["train", "validation", "test"]:
        features = pd.read_pickle(prefix + f"dataset.lemma_avg_v3.seqs.{l}")
        w2v_features.append(np.array(features))
    
    return tuple(w2v_features)