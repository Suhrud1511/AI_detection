# main.py
from feature_extraction import build_feature_dataframe, detect_word_et, detect_word_others_researchers, \
    detect_word_this, detect_word_because, detect_word_but, detect_word_however, detect_word_alhough, \
    detect_capital_letters_vs_periods, detect_numbers_per_paragraph, detect_long_sentences_per_paragraph, \
    detect_short_sentences_per_paragraph, mean_diff_in_sentence_length_per_paragraph, std_dev_sentence_length_per_paragraph, \
    detect_apostrophe_per_paragraph, detect_question_mark_per_paragraph, detect_semicolon_colon_per_paragraph, \
    detect_dash_per_paragraph, detect_parentheses_per_paragraph, count_words_per_paragraph, count_sentences_per_paragraph
from model_training import train_xgboost_model, train_adaboost_model, evaluate_model, predict_submission, plot_roc_auc
from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
app = Flask(__name__)

import pickle

# Load the XGBoost model
with open('xgboost_model.pkl', 'rb') as file:
    xgboost_model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_input_features = build_feature_dataframe([user_input], [0])

        xgboost_prediction = predict_submission(xgboost_model, user_input_features.drop('label', axis=1))
        # adaboost_prediction = predict_submission(adaboost_model, user_input_features.drop('label', axis=1))


        return render_template('result.html', prediction=xgboost_prediction*100)

if __name__ == "__main__":
    app.run(debug=True)