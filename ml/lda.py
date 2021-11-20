import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

## adapted from Daniel's codes

def build_count_vectorizer(all_text):
    vectorizer = CountVectorizer(analyzer='word',
                            strip_accents='unicode',
                            min_df=0,                        # minimum reqd occurences of a word 
                            stop_words='english',             # remove stop words
                            lowercase=True,                   # convert all words to lowercase
                            max_features=15000,             # max number of uniq words
                            token_pattern='\\w{1,}',
                            ngram_range=(1, 1),
                            )

    return vectorizer.fit(all_text)

def get_bow_vect(train_text, val_text):
    bow_vectorizer = build_count_vectorizer(pd.concat([train_text, val_text]))
    X_train_lda = bow_vectorizer.transform(train_text)
    X_val_lda = bow_vectorizer.transform(val_text)
    
    return X_train_lda, X_val_lda, bow_vectorizer

## referenced from 
### https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
### https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#7createthedocumentwordmatrix

# build LDA based on bow
def generate_lda_features(X_train_lda, X_val_lda, n_topics):

    model = LatentDirichletAllocation(
        n_components = n_topics,
        max_iter = 10,
        learning_method = "online",
        random_state = 42,
        n_jobs = 1
    )
    
    lda_model = model.fit(vstack([X_train_lda, X_val_lda]))
    X_train_lda = lda_model.transform(X_train_lda)
    X_val_lda = lda_model.transform(X_val_lda)

    return X_train_lda, X_val_lda, lda_model

def print_top_words_per_topic(model, feature_names):

    all_top_20_words = []
    for i, topic in enumerate(model.components_):
        words_wgt = list(zip(feature_names, topic))
        top_20_important = sorted(words_wgt, key=lambda x: x[1], reverse = True)[:20]
        top_20_important = [x for x,y in top_20_important]
        print(f"LDA_{i}:\n{' '.join(top_20_important)}\n")
        all_top_20_words.append(top_20_important)

    return all_top_20_words