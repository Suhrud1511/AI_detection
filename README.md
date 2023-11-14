# ML_MiniProj
# Essay Classification Project

## Overview

The **Essay Classification Project** is a machine learning endeavor designed to categorize essays into specific genres or prompts. The project employs feature extraction techniques and two distinct models—XGBoost and AdaBoost with a logistic regression base estimator—to achieve accurate essay classification.

### Key Components

- **Feature Extraction:** The `feature_extraction.py` module utilizes various functions to extract essential features from essay texts. These features include sentence and word counts, punctuation presence, and the identification of specific words or phrases.

- **Model Training:** The `model_training.py` module is responsible for training machine learning models. It employs XGBoost and AdaBoost with a logistic regression base estimator. The ROC AUC scores are evaluated and presented in a bar graph for model performance comparison.

- **Data Handling:** The `data/` directory stores input data, exemplified by `train_essays_7_prompts.csv`. The `output/` directory contains the generated submission file, `submission.csv`, which includes the average predictions of both the XGBoost and AdaBoost models.

### Usage

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## References
This is an Implementation of the conclusions from the paper cited below

- Author(s)  Heather Desaire,Aleesa E. Chua,Min-Gyu Kim,David Hua . "Accurately detecting AI text when ChatGPT is told to write like a chemist." *Elsevier*, 2023, [DOI or Link](https://www.sciencedirect.com/science/article/pii/S2666386423005015#bib7).