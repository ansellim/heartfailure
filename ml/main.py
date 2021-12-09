"""
Main script to run all :
1. Preprocessing
2. Feature engineeering 
3. Train Classification Models (xgboost and SVM)
4. Evaluate  Models 
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from feature_engineering import pre_process_text, get_data, build_tfidf_vectorizer
from hotspot import get_hotspot_feature_names, generate_hotspot_features, plot_features
from lda import get_bow_vect, generate_lda_features, print_top_words_per_topic
from word2vec import generate_w2v_features
from model import run_xgboost, true_evaluation
from datetime import datetime
from scipy.sparse import hstack, csr_matrix
from svm_classifier import run_svm

def min_max_norm(mat):
    """
    Min-max scaling of data before feature engineering process
    """
    mat = mat.todense()
    scaler = MinMaxScaler()
    scaler.fit(mat)
    mat = scaler.transform(mat)
    return csr_matrix(mat)

def get_all_data():
    """
    wrapper to get all raw data inputs
    """    
    train = get_data('train')
    test = get_data('test')
    val = get_data('val')

    return train, test, val

def ml_pipeline(train, test, val, VECTORISER_MAX_FEATURES, N_TOPICS, XGB_ETA, XGB_EARLY_STOP):
    """
    wrapper function to run entire classic ML pipeline with both XGBoost and SVM
    """

    # TF-IDF Feature engineering
    all_text = pd.concat([train['text'],val['text']])
    tfidf_vectorizer = build_tfidf_vectorizer(all_text, VECTORISER_MAX_FEATURES)
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
    X_train_lda, X_val_lda, bow_vectorizer = get_bow_vect(train["text"], val["text"], VECTORISER_MAX_FEATURES)
    X_train_lda, X_val_lda, lda_model = generate_lda_features(X_train_lda, X_val_lda, N_TOPICS)
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

    all_feature_names = tfvocab + hp_feature_names + [f"LDA_{i}" for i in range(N_TOPICS)] + [f"w2v_{i}" for i in range(100)]

    d_train = xgb.DMatrix(X_train, y_train,feature_names=all_feature_names)
    d_val = xgb.DMatrix(X_val,y_val,feature_names=all_feature_names)

    xgboost_model = run_xgboost(d_train,d_val, XGB_ETA, XGB_EARLY_STOP)

    # Run true evaluation on out-of-sample (test) set
    X_test_tfidf = tfidf_vectorizer.transform(test['text'])
    X_test_hotspot = generate_hotspot_features(test["text"], hf_terms)
    
    X_test_lda = bow_vectorizer.transform(test["text"])
    X_test_lda = lda_model.transform(X_test_lda)

    X_test = hstack([X_test_tfidf, X_test_hotspot, X_test_lda, X_test_w2v])
    X_test = min_max_norm(X_test)

    y_true = test['y']

    d_test = xgb.DMatrix(X_test,feature_names=all_feature_names)
    xgb_acc, xgb_auc, xgb_f1 = true_evaluation(xgboost_model, d_test, y_true)

    svm_acc, svm_auc, svm_f1 = run_svm(X_train, y_train, X_val, y_val, X_test, y_true)

    plot_features(X_train.todense(), "X_train")
    plot_features(X_val.todense(), "X_val")
    plot_features(X_test.todense(), "X_test")

    return xgb_acc, xgb_auc, xgb_f1, svm_acc, svm_auc, svm_f1


if __name__ == "__main__":
    start = datetime.now()
    print(f'Script start: {start}\n')

    # Raw data inputs
    train, test, val = get_all_data()

    ## initiate values for hyperparameter tuning
    INIT_VECTORISER_MAX_FEATURES = list(range(10000, 25001, 5000))
    INIT_N_TOPICS = list(range(5,21,5))

    INIT_XGB_ETA = np.linspace(0.1, 0.4, 4)
    INIT_XGB_EARLY_STOP = list(range(5,21,5))

    tuning_results = []
    for i in range(4):

        res = []

        VECTORISER_MAX_FEATURES = INIT_VECTORISER_MAX_FEATURES[i]
        N_TOPICS = INIT_N_TOPICS[i]
        XGB_ETA = INIT_XGB_ETA[i]
        XGB_EARLY_STOP = INIT_XGB_EARLY_STOP[i]

        res.extend([VECTORISER_MAX_FEATURES, N_TOPICS, XGB_ETA, XGB_EARLY_STOP])
        xgb_acc, xgb_auc, xgb_f1, svm_acc, svm_auc, svm_f1 = ml_pipeline(train, test, val, VECTORISER_MAX_FEATURES, N_TOPICS, XGB_ETA, XGB_EARLY_STOP)
        res.extend([xgb_acc, xgb_auc, xgb_f1, svm_acc, svm_auc, svm_f1])
        
        tuning_results.append(res)

    tuning_results = pd.DataFrame(tuning_results, columns=["Max Number of Features in Vectorizer", "Number of LDA Topics", "XGBoost ETA", "XGBoost Early Stop",\
                                                "XGBoost Accuracy", "XGBoost AUC", "XGBoost F1-Score", \
                                                "SVM Accuracy", "SVM AUC", "SVM F1-Score"])
    tuning_results.to_csv("tuning_results.csv")

    print("\n\n\n======== Hyperparameter Tuning Results =========")
    print(tuning_results.to_string())
    print("\n\n\n")

    ## get best parameters
    argmax = np.argmax(tuning_results.iloc[:, 4:].mean(axis = 1))
    VECTORISER_MAX_FEATURES, N_TOPICS, XGB_ETA, XGB_EARLY_STOP = tuning_results.loc[argmax, ["Max Number of Features in Vectorizer", "Number of LDA Topics", "XGBoost ETA", "XGBoost Early Stop"]].values.tolist()

    ## run pipeline with best parameters
    _ = ml_pipeline(train, test, val, int(VECTORISER_MAX_FEATURES), int(N_TOPICS), XGB_ETA, int(XGB_EARLY_STOP))

    end = datetime.now()
    print(f'Script end: {end}')
    print(f'Script Runtime:{end - start}\n')


