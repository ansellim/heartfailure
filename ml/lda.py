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