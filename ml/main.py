import pandas as pd
import xgboost as xgb
from feature_engineering import pre_process_text, get_data, build_tfidf_vectorizer
from hotspot import get_hotspot_feature_names, generate_hotspot_features
from lda import build_count_vectorizer, generate_lda_features
# from word2vec import generate_w2v_features
from model import run_xgboost, true_evaluation
from datetime import datetime
from scipy.sparse import hstack

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
    bow_vectorizer = build_count_vectorizer(all_text)
    X_train_lda = bow_vectorizer.transform(train["text"])
    X_val_lda = bow_vectorizer.transform(val["text"])

    n_topics = 10
    X_train_lda = generate_lda_features(X_train_lda, n_topics)
    X_val_lda = generate_lda_features(X_val_lda, n_topics)

    # word2vec feature engineering
    # X_train_w2v = generate_w2v_features(train["text"])
    # X_val_w2v = generate_w2v_features(val["text"])

    # Train XGB model
    X_train = hstack([X_train_tfidf, X_train_hotspot, X_train_lda])
    X_val = hstack([X_val_tfidf, X_val_hotspot, X_val_lda])

    y_train = train['y']
    y_val = val['y']

    all_feature_names = tfvocab + hp_feature_names + [f"LDA_{i}" for i in range(n_topics)]

    d_train = xgb.DMatrix(X_train, y_train,feature_names=all_feature_names)
    d_val = xgb.DMatrix(X_val,y_val,feature_names=all_feature_names)

    model = run_xgboost(d_train,d_val)

    # Run true evaluation on out-of-sample (test) set
    X_test_tfidf = tfidf_vectorizer.transform(test['text'])
    X_test_hotspot = generate_hotspot_features(test["text"], hf_terms)
    
    X_test_lda = bow_vectorizer.transform(test["text"])
    X_test_lda = generate_lda_features(X_test_lda, n_topics)

    X_test = hstack([X_test_tfidf, X_test_hotspot, X_test_lda])

    y_true = test['y']

    d_test = xgb.DMatrix(X_test,feature_names=all_feature_names)
    true_evaluation(model, d_test, y_true)

    end = datetime.now()
    print(f'Script end: {end}')
    print(f'Script Runtime:{end - start}\n')


