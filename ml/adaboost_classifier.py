'''
Ansel Lim, 13 Nov 2021

Basic Adaboost classifier

'''

from datetime import datetime
import joblib
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from feature_engineering import get_data
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Timestamp / start script
start = datetime.now()
print(f'Script start: {start}\n')

# Raw data inputs
train = get_data('train')
test = get_data('test')
val = get_data('val')

train_and_val = pd.concat([train,val])

# Fit the model
clf = Pipeline([('vectorizer', CountVectorizer(stop_words='english')),
                ('tf_idf', TfidfTransformer()),
                ('clf', AdaBoostClassifier(base_estimator= DecisionTreeClassifier(),n_estimators=500, random_state=42))
                ])

# https://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier
parameters = {'clf__base_estimator__max_depth':[i for i in range(2,11,2)],
              'clf__base_estimator__min_samples_leaf':[5,10],
              'clf__n_estimators':[10,50,250,1000],
              'clf__learning_rate':[0.001,0.01,0.1]}

grid_clf = GridSearchCV(clf,parameters)

grid_clf.fit(train_and_val.text,train_and_val.y)

# Display test score for optimized parameters
grid_clf.best_estimator_.score(test.text,test.y)

# Save the best model
joblib.dump(grid_clf.best_estimator_,"./models/adaboost_classifier_" + str(start.strftime("%Y%m%d_%H%M%S")) + "_.pkl")