'''
Ansel Lim, 13 Nov 2021

Basic SVM classifier with a little bit of grid search. This is very rough work.

With reference to https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
'''

from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def run_svm(X_train, y_train, X_val, y_val, X_test, y_test):
    # Fit the model
    base_clf = Pipeline([('sgd',SGDClassifier(loss='hinge',
                                        penalty='l2',
                                        alpha=0.0001,
                                        max_iter=1000,
                                        random_state=42))])
    params = {'sgd__alpha': (1e-2, 1e-3, 1e-4)}
    grid_search = GridSearchCV(base_clf,params,n_jobs=-1)
    grid_search.fit(X_train,y_train)

    print("\n\n\n----------Training SVM---------")

    # Display best score / best params
    print("Best Score: ", grid_search.best_score_)
    print("Best Parameters: ",grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Check on val & test data
    print("Val data", best_model.score(X_val, y_val))
    print("Test data", best_model.score(X_test, y_test))
    
    pred_test = best_model.predict(X_test)
    print("\n----------Best SVM---------")
    print(f"Accuracy: {accuracy_score(y_test, pred_test)}")
    print(f"AUC: {roc_auc_score(pred_test, y_test)}")
    print(f"F1-score: {f1_score(y_test, pred_test)}")

    # Save the best model
    joblib.dump(best_model,"./models/svm_classifier_" + str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "_.pkl")

    return best_model