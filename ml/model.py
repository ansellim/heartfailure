import xgboost as xgb
from xgboost.core import DMatrix
from xgboost.sklearn import XGBClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_curve, accuracy_score

def run_xgboost(d_train:DMatrix, d_val:DMatrix) -> XGBClassifier:
    """
    Run xgboost model using training and validation dataset
    Print out best AUC reached and save feature importance plot

    Output: Trained xgb model
    """
    # Modeling
    xgb_params = {
        'eta': 0.3, 
        'max_depth': 5, 
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
        'objective': 'binary:logistic', 
        'eval_metric': 'auc'
    }

    print('Training XGB model...')
    model = xgb.train(
        params = xgb_params,
        dtrain=d_train, 
        num_boost_round=100,
        evals=[(d_val, 'valid')],
        verbose_eval=True, 
        early_stopping_rounds=10)
    
    # Training outputs
    print(f'\n----------Training Best performance----------')
    print(model.attributes())
    xgb.plot_importance(model, max_num_features=25)
    plt.savefig('xgboost_plot_importance.png',bbox_inches='tight',dpi = 300)

    return model

def true_evaluation(model:XGBClassifier, d_test:DMatrix, y_true:pd.Series):
    """
    Run model's predictions against out-of-sample (test) set
    And report on full metrics table and plot AUC-ROC curve
    """
    y_pred = model.predict(d_test)

    # Convert predictions into hard labels
    y_hard = y_pred
    y_hard[y_hard > 0.5] = 1
    y_hard[y_hard <= 0.5] = 0

    # Print classification report
    print('\n----------Classification Report---------')
    print(classification_report(y_true, y_hard, target_names=['Non-HF', 'HF']))

    # Print confusion matrix
    print('\n----------Confusion Matrix---------')
    print(confusion_matrix(y_true, y_pred))

    print('\n----------Evaluation Metrics---------')
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    print(f'Accuracy: {accuracy_score(y_true,y_pred)} | AUC:{auc(fpr, tpr)}')
