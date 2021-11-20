import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import pre_process_text, get_data, build_tfidf_vectorizer
from hotspot import get_hotspot_feature_names, generate_hotspot_features, plot_features
from lda import get_bow_vect, generate_lda_features, print_top_words_per_topic
from word2vec import generate_w2v_features
from model import run_xgboost, true_evaluation
from datetime import datetime
from scipy.sparse import hstack, csr_matrix

def min_max_norm(mat):
    mat = mat.todense()
    scaler = MinMaxScaler()
    scaler.fit(mat)
    mat = scaler.transform(mat)
    return csr_matrix(mat)

if __name__ == "__main__":
    start = datetime.now()
    print(f'Script start: {start}\n')

    # Raw data inputs
    train = get_data('train')
    test = get_data('test')
    val = get_data('val')

    # TF-IDF Feature engineering
    all_text = pd.concat([train['text'],val['text']])
    tfidf_vectorizer = build_tfidf_vectorizer(all_text)
    tfvocab = tfidf_vectorizer.get_feature_names()

    X_train_tfidf = tfidf_vectorizer.transform(train['text'])
    print(f'X_train_tfidf shape:{X_train_tfidf.shape}')
    
    X_val_tfidf = tfidf_vectorizer.transform(val['text'])
    print(f'X_val_tfidf shape:{X_val_tfidf.shape}')
    
    # Hotspot technique feature engineering
    hf_terms, hp_feature_names = get_hotspot_feature_names()
    X_train_hotspot = generate_hotspot_features(train["text"], hf_terms)
    X_val_hotspot = generate_hotspot_features(val["text"], hf_terms)

    # LDA feature engineering from BOW
    n_topics = 10
    X_train_lda, X_val_lda, bow_vectorizer = get_bow_vect(train["text"], val["text"])
    X_train_lda, X_val_lda, lda_model = generate_lda_features(X_train_lda, X_val_lda, n_topics)
    all_top_20 = print_top_words_per_topic(lda_model, bow_vectorizer.get_feature_names())

    # word2vec feature engineering
    X_train_w2v, X_val_w2v, X_test_w2v = generate_w2v_features()

    # Train XGB model
    X_train = hstack([X_train_tfidf, X_train_hotspot, X_train_lda, X_train_w2v])
    X_train = min_max_norm(X_train)
    X_val = hstack([X_val_tfidf, X_val_hotspot, X_val_lda, X_val_w2v])
    X_val = min_max_norm(X_val)

    y_train = train['y']
    y_val = val['y']

    all_feature_names = tfvocab + hp_feature_names + [f"LDA_{i}" for i in range(n_topics)] + [f"w2v_{i}" for i in range(100)]

    d_train = xgb.DMatrix(X_train, y_train,feature_names=all_feature_names)
    d_val = xgb.DMatrix(X_val,y_val,feature_names=all_feature_names)

    model = run_xgboost(d_train,d_val)

    # Run true evaluation on out-of-sample (test) set
    X_test_tfidf = tfidf_vectorizer.transform(test['text'])
    X_test_hotspot = generate_hotspot_features(test["text"], hf_terms)
    
    X_test_lda = bow_vectorizer.transform(test["text"])
    X_test_lda = lda_model.transform(X_test_lda)

    X_test = hstack([X_test_tfidf, X_test_hotspot, X_test_lda, X_test_w2v])
    X_test = min_max_norm(X_test)

    y_true = test['y']

    d_test = xgb.DMatrix(X_test,feature_names=all_feature_names)
    true_evaluation(model, d_test, y_true)

    plot_features(X_train.todense(), "X_train")
    plot_features(X_val.todense(), "X_val")
    plot_features(X_test.todense(), "X_test")

    end = datetime.now()
    print(f'Script end: {end}')
    print(f'Script Runtime:{end - start}\n')


