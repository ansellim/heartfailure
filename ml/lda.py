## referenced from 
### https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
### https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#7createthedocumentwordmatrix

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

import gensim

def gen_bag_of_words(data):
    # Create Dictionary
    id2word = corpora.Dictionary(data)
    # Term Document Frequency
    bow = [id2word.doc2bow(text) for text in data]

    return id2word, bow

def gen_lda_model(id2word, bow):
    model = gensim.models.LdaMulticore(
            bow, 
            num_topics = 10,
            id2word = id2word,
            passes = 2, 
            workers = 2)
    return model

def generate_lda_features(data):
    id2word, bow = gen_bag_of_words(data)
    model = gen_lda_model(id2word, bow)

    print("DONE")
