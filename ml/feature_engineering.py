import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def pre_process_text(clean_text:str) -> str:
    """
    Pre process raw text by applying
    - lemmatizing
    - stopword removal
    - regex removal of punctuation
    - lowercase
    - join all words into 1 long string
    """
    lemmatizer = WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = clean_text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [t for t in text if t not in set(stopwords)]
    text = [lemmatizer.lemmatize(t) for t in text]
    text = ' '.join(text)

    return text

def get_data(type:str) -> pd.DataFrame:
    """
    Reads raw data from data.pkl, pre-process 
    and returns train/test/val dataframe
    """
    with open('./datasets/data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    text= data[f'{type}_texts']
    y =data[f'{type}_labels']

    df = pd.DataFrame({'text':text,'y':y})
    df['text'] = df['text'].apply(pre_process_text)

    print(f'{type} df loaded, Shape:{df.shape}')

    return df

def build_tfidf_vectorizer(all_text:pd.DataFrame) -> TfidfVectorizer:
    """
    Inputs: A dataframe of all text from train and test sets (corpus)
    Output: A TF-IDF word vectorizer trained the given corpus
    """
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 1),
        norm='l2',
        min_df=0,
        smooth_idf=False,
        max_features=15000)
    word_vectorizer.fit(all_text)
    
    return word_vectorizer

## referenced from 
### https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
### https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#7createthedocumentwordmatrix

# build LDA based on tfidf
def generate_lda_features(data, n_topics):
    model = LatentDirichletAllocation(
        n_components = n_topics,
        max_iter = 10,
        learning_method = "online",
        random_state = 42,
        n_jobs = 1
    )
    output = model.fit_transform(data)
    return output