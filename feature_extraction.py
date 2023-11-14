# feature_extraction.py
import re
import statistics
import pandas as pd
def count_sentences_per_paragraph(text):
    paragraphs = text.split('\n\n')
    paragraphs = [p for p in paragraphs if p.strip()]

    sentence_counts = []
    for paragraph in paragraphs:
        sentences = re.split(r'[.?!]+', paragraph)
        sentences = [s for s in sentences if s.strip()]
        sentence_counts.append(len(sentences))

    return sentence_counts

def count_words_per_paragraph(text):
    paragraphs = text.split('\n\n')
    paragraphs = [p for p in paragraphs if p.strip()]

    word_counts = []
    for paragraph in paragraphs:
        words = paragraph.split()
        word_counts.append(len(words))

    return word_counts

def detect_parentheses_per_paragraph(text):
    paragraphs = text.split('\n\n')

    parentheses_presence = []
    for paragraph in paragraphs:
        if '(' in paragraph or ')' in paragraph:
            parentheses_presence.append(1)
        else:
            parentheses_presence.append(0)

    return parentheses_presence

def detect_dash_per_paragraph(text):
    paragraphs = text.split('\n\n')

    dash_presence = []
    for paragraph in paragraphs:
        if '-' in paragraph:
            dash_presence.append(1)
        else:
            dash_presence.append(0)

    return dash_presence

def detect_semicolon_colon_per_paragraph(text):
    paragraphs = text.split('\n\n')

    punctuation_presence = []
    for paragraph in paragraphs:
        if ';' in paragraph or ':' in paragraph:
            punctuation_presence.append(1)
        else:
            punctuation_presence.append(0)

    return punctuation_presence

def detect_question_mark_per_paragraph(text):
    paragraphs = text.split('\n\n')

    question_mark_presence = []
    for paragraph in paragraphs:
        if '?' in paragraph:
            question_mark_presence.append(1)
        else:
            question_mark_presence.append(0)

    return question_mark_presence

def detect_apostrophe_per_paragraph(text):
    paragraphs = text.split('\n\n')

    apostrophe_presence = []
    for paragraph in paragraphs:
        if "'" in paragraph:
            apostrophe_presence.append(1)
        else:
            apostrophe_presence.append(0)

    return apostrophe_presence

def std_dev_sentence_length_per_paragraph(text):
    paragraphs = text.split('\n\n')

    std_devs = []
    for paragraph in paragraphs:
        sentences = re.split(r'[.?!]+', paragraph)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]

        if len(sentence_lengths) > 1:
            std_dev = statistics.stdev(sentence_lengths)
        else:
            std_dev = 0
        std_devs.append(std_dev)

    return std_devs

def mean_diff_in_sentence_length_per_paragraph(text):
    paragraphs = text.split('\n\n')

    mean_diffs = []
    for paragraph in paragraphs:
        sentences = re.split(r'[.?!]+', paragraph)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        differences = [abs(sentence_lengths[i] - sentence_lengths[i+1]) for i in range(len(sentence_lengths) - 1)]

        if differences:
            mean_diff = statistics.mean(differences)
        else:
            mean_diff = 0
        mean_diffs.append(mean_diff)

    return mean_diffs

def detect_short_sentences_per_paragraph(text):
    paragraphs = text.split('\n\n')

    short_sentence_presence = []
    for paragraph in paragraphs:
        sentences = re.split(r'[.?!]+', paragraph)
        has_short_sentence = any(len(sentence.split()) < 11 for sentence in sentences if sentence.strip())
        short_sentence_presence.append(1 if has_short_sentence else 0)

    return short_sentence_presence

def detect_long_sentences_per_paragraph(text):
    paragraphs = text.split('\n\n')

    long_sentence_presence = []
    for paragraph in paragraphs:
        sentences = re.split(r'[.?!]+', paragraph)
        has_long_sentence = any(len(sentence.split()) > 34 for sentence in sentences if sentence.strip())
        long_sentence_presence.append(1 if has_long_sentence else 0)

    return long_sentence_presence

def detect_numbers_per_paragraph(text):
    paragraphs = text.split('\n\n')

    number_presence = []
    for paragraph in paragraphs:
        has_number = any(char.isdigit() for char in paragraph)
        number_presence.append(1 if has_number else 0)

    return number_presence

def detect_capital_letters_vs_periods(text):
    paragraphs = text.split('\n\n')

    capital_vs_period_presence = []
    for paragraph in paragraphs:
        capital_letters = sum(1 for char in paragraph if char.isupper())
        periods = paragraph.count('.')

        if capital_letters >= 2 * periods:
            capital_vs_period_presence.append(1)
        else:
            capital_vs_period_presence.append(0)

    return capital_vs_period_presence

def detect_word_alhough(text):
    paragraphs = text.split('\n\n')
    presence = [1 if 'although' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_however(text):
    paragraphs = text.split('\n\n')
    presence = [1 if 'however' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_but(text):
    paragraphs = text.split('\n\n')
    presence = [1 if ' but ' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_because(text):
    paragraphs = text.split('\n\n')
    presence = [1 if 'because' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_this(text):
    paragraphs = text.split('\n\n')
    presence = [1 if 'this' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_others_researchers(text):
    paragraphs = text.split('\n\n')
    presence = [1 if 'others' in paragraph.lower() or 'researchers' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence

def detect_word_et(text):
    paragraphs = text.split('\n\n')
    presence = [1 if ' et ' in paragraph.lower() else 0 for paragraph in paragraphs]
    return presence
def build_feature_dataframe(texts, labels):
    results = []
    for text in texts:
        result = {
            'sentences_per_paragraph': count_sentences_per_paragraph(text),
            'words_per_paragraph': count_words_per_paragraph(text),
            'parentheses_presence': detect_parentheses_per_paragraph(text),
            'dash_presence': detect_dash_per_paragraph(text),
            'semicolon_colon_presence': detect_semicolon_colon_per_paragraph(text),
            'question_mark_presence': detect_question_mark_per_paragraph(text),
            'apostrophe_presence': detect_apostrophe_per_paragraph(text),
            'std_dev_sentence_length': std_dev_sentence_length_per_paragraph(text),
            'mean_diff_consecutive_sentences': mean_diff_in_sentence_length_per_paragraph(text),
            'short_sentence_presence': detect_short_sentences_per_paragraph(text),
            'long_sentence_presence': detect_long_sentences_per_paragraph(text),
            'number_presence': detect_numbers_per_paragraph(text),
            'capital_vs_period_presence': detect_capital_letters_vs_periods(text),
            'word_alhough_presence': detect_word_alhough(text),
            'word_however_presence': detect_word_however(text),
            'word_but_presence': detect_word_but(text),
            'word_because_presence': detect_word_because(text),
            'word_this_presence': detect_word_this(text),
            'word_others_researchers_presence': detect_word_others_researchers(text),
            'word_et_presence': detect_word_et(text),
        }
        results.append(result)

    rows = []
    for text_id, (result, label) in enumerate(zip(results, labels)):
        for i in range(len(result['sentences_per_paragraph'])):
            row = {key: value[i] if isinstance(value, list) and len(value) > i else value for key, value in result.items()}
            row['text_id'] = text_id
            row['label'] = label
            rows.append(row)

    return pd.DataFrame(rows)