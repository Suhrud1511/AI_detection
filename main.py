# main.py
from feature_extraction import build_feature_dataframe, detect_word_et, detect_word_others_researchers, \
    detect_word_this, detect_word_because, detect_word_but, detect_word_however, detect_word_alhough, \
    detect_capital_letters_vs_periods, detect_numbers_per_paragraph, detect_long_sentences_per_paragraph, \
    detect_short_sentences_per_paragraph, mean_diff_in_sentence_length_per_paragraph, std_dev_sentence_length_per_paragraph, \
    detect_apostrophe_per_paragraph, detect_question_mark_per_paragraph, detect_semicolon_colon_per_paragraph, \
    detect_dash_per_paragraph, detect_parentheses_per_paragraph, count_words_per_paragraph, count_sentences_per_paragraph
from model_training import train_xgboost_model, train_adaboost_model, evaluate_model, predict_submission, plot_roc_auc

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('train_essays_7_prompts.csv')
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train_features = build_feature_dataframe(df_train['text'].tolist(), df_train['label'].tolist())
    df_test_features = build_feature_dataframe(df_test['text'].tolist(), [0] * len(df_test))  # Provide dummy labels for testing data

    X_train, X_test, y_train, y_test = train_test_split(df_train_features.drop('label', axis=1),
                                                        df_train_features['label'], test_size=0.2, random_state=42)

    xgboost_model = train_xgboost_model(X_train, y_train)
    adaboost_model = train_adaboost_model(X_train, y_train)

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
