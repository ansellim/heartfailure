import pandas as pd
import xgboost as xgb
from feature_engineering import pre_process_text, get_data, build_tfidf_vectorizer
from model import run_xgboost, true_evaluation
from datetime import datetime


if __name__ == "__main__":
    start = datetime.now()
    print(f'Script start: {start}\n')

    # Raw data inputs
    train = get_data('train')
    test = get_data('test')
    val = get_data('val')

    # TF-IDF Feature engineering
    all_text = pd.concat([train['text'],test['text']])
    vectorizer = build_tfidf_vectorizer(all_text)
    tfvocab = vectorizer.get_feature_names()

    # Train XGB model
    X_train = vectorizer.transform(train['text'])
    print(f'X_train shape:{X_train.shape}')
    y_train = train['y']
    d_train = xgb.DMatrix(X_train, y_train,feature_names=tfvocab)

    X_val = vectorizer.transform(val['text'])
    print(f'X_val shape:{X_val.shape}')
    y_val = val['y']
    d_val = xgb.DMatrix(X_val,y_val,feature_names=tfvocab)

    model = run_xgboost(d_train,d_val)

    # Run true evaluation on out-of-sample (test) set
    X_test = vectorizer.transform(val['text'])
    y_true = val['y']
    d_test = xgb.DMatrix(X_test,feature_names=tfvocab)
    true_evaluation(model, d_val, y_true)

    end = datetime.now()
    print(f'Script end: {end}')
    print(f'Script Runtime:{end - start}\n')


