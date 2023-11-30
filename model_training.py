# model_training.py
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from feature_extraction import build_feature_dataframe, detect_word_et, detect_word_others_researchers, \
    detect_word_this, detect_word_because, detect_word_but, detect_word_however, detect_word_alhough, \
    detect_capital_letters_vs_periods, detect_numbers_per_paragraph, detect_long_sentences_per_paragraph, \
    detect_short_sentences_per_paragraph, mean_diff_in_sentence_length_per_paragraph, std_dev_sentence_length_per_paragraph, \
    detect_apostrophe_per_paragraph, detect_question_mark_per_paragraph, detect_semicolon_colon_per_paragraph, \
    detect_dash_per_paragraph, detect_parentheses_per_paragraph, count_words_per_paragraph, count_sentences_per_paragraph
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier

def train_xgboost_model(X_train, y_train):
   
    param_grid = {
        'max_depth': [3, 4, 5],
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.2, 0.3],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc')

    
    xgb_random = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_grid, n_iter=10,
                                     scoring='roc_auc', n_jobs=-1, cv=5, random_state=42)

    xgb_random.fit(X_train, y_train)
    best_params = xgb_random.best_params_

    # Use the best parameters to train the final model
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    return final_model



def train_adaboost_model(X_train, y_train):
   
    param_grid = {
        'base_estimator__max_depth': [2, 3, 4],
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.1, 0.2, 0.3]
    }

    base_estimator = DecisionTreeClassifier()

    adaboost_model = AdaBoostClassifier(base_estimator=base_estimator, algorithm='SAMME.R')

   
    adaboost_random = RandomizedSearchCV(estimator=adaboost_model, param_distributions=param_grid, n_iter=10,
                                         scoring='roc_auc', n_jobs=-1, cv=5, random_state=42)
    
    adaboost_random.fit(X_train, y_train)
    best_params = adaboost_random.best_params_
   
    best_base_estimator = DecisionTreeClassifier(max_depth=best_params['base_estimator__max_depth'])
    adaboost_model = AdaBoostClassifier(base_estimator=best_base_estimator, n_estimators=best_params['n_estimators'],
                                        learning_rate=best_params['learning_rate'], algorithm='SAMME.R')

    # Train the final AdaBoost model
    adaboost_model.fit(X_train, y_train)

    return adaboost_model

 


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_pred[:, 1])
    return auc

def predict_submission(model, X_test):
    preds = model.predict_proba(X_test)[:, 1]
    return preds

def plot_roc_auc(fpr, tpr, auc, label):
    plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.2f})')

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    df = pd.read_csv("train_essays_7_prompts_v2.csv")
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train_features = build_feature_dataframe(df_train['text'].tolist(), df_train['label'].tolist())
    df_test_features = build_feature_dataframe(df_test['text'].tolist(), [0] * len(df_test))  # Provide dummy labels for testing data

    X_train, X_test, y_train, y_test = train_test_split(df_train_features.drop('label', axis=1),
                                                        df_train_features['label'], test_size=0.2, random_state=42)

    xgboost_model = train_xgboost_model(X_train, y_train)
    adaboost_model = train_adaboost_model(X_train, y_train)
    save_model(xgboost_model, 'xgboost_model.pkl')
    xgboost_auc = evaluate_model(xgboost_model, X_test, y_test)
    adaboost_auc = evaluate_model(adaboost_model, X_test, y_test)

    print("XGBoost ROC AUC:", xgboost_auc)
    print("AdaBoost ROC AUC:", adaboost_auc)

    xgboost_preds = predict_submission(xgboost_model, df_test_features.drop('label', axis=1))
    adaboost_preds = predict_submission(adaboost_model, df_test_features.drop('label', axis=1))

    df_test_features['xgboost_generated'] = xgboost_preds
    df_test_features['adaboost_generated'] = adaboost_preds

    average_predictions = df_test_features.groupby(['text_id'])[['xgboost_generated', 'adaboost_generated']].mean().reset_index()
    average_predictions.to_csv('submission.csv', index=False)

    # Bar graph for ROC AUC scores
    labels = ['XGBoost', 'AdaBoost']
    scores = [xgboost_auc, adaboost_auc]

    plt.bar(labels, scores, color=['blue', 'orange'])
    plt.ylabel('ROC AUC Score')
    plt.title('Comparison of ROC AUC Scores')
    plt.show()

if __name__ == "__main__":
    main()
