'''
Ansel Lim, 13 Nov 2021

Basic SVM classifier with a little bit of grid search. This is very rough work.

With reference to https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
'''

from feature_engineering import get_data
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import joblib

# Start script
start = datetime.now()
print(f'Script start: {start}\n')

# Raw data inputs
train = get_data('train')
test = get_data('test')
val = get_data('val')

# Fit the model
base_clf = Pipeline([('vectorizer',CountVectorizer(stop_words='english')),
                    ('tf_idf',TfidfTransformer()),
                    ('sgd',SGDClassifier(loss='hinge',
                                    penalty='l2',
                                    alpha=0.0001,
                                    max_iter=1000,
                                    random_state=42))])
params = {'vectorizer__ngram_range': [(1, 1), (1, 2), (2,2)], 'tf_idf__use_idf': (True, False), 'sgd__alpha': (1e-2, 1e-3, 1e-4)}
grid_search = GridSearchCV(base_clf,params,n_jobs=-1)
grid_search.fit(train.text,train.y)

# Display best score / best params
print(grid_search.best_score_)
print(grid_search.best_params_)

# Check on val & test data
print("Val data", grid_search.best_estimator_.score(val.text,val.y))
print("Test data", grid_search.best_estimator_.score(test.text,test.y))

# Save the best model
joblib.dump(grid_search.best_estimator_,"./models/svm_classifier_" + str(start) + "_.pkl")

