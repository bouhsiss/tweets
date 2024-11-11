# tweets
the goal of this project is to give a first approach to NLP, where we'll preprocess text data and train different classifiers trying to solve a classification task with the best possible score.

# Overview
This repository contains a sentiment analysis project that utilizes various natural language processing (NLP) techniques to analyze sentiments in tweets. The project is structured into different modules for text preprocessing, vectorization, and model training. The main functionality is encapsulated in a Jupyter Notebook that orchestrates the entire workflow.

## Files

### 1. `notebooks/sentiment_analysis.ipynb`
This Jupyter Notebook serves as the main entry point for the sentiment analysis project. It includes the following key steps:
- **Data Loading**: Loads datasets containing tweets with different sentiments (positive, negative, neutral).
- **Data Preprocessing**: Cleans the data by removing duplicates and stop words, and transforms the text to lowercase.
- **Data Splitting**: Splits the dataset into training and testing sets while maintaining the sentiment distribution.
- **Text Preprocessing and Vectorization**: Prepares the text data using various preprocessing techniques (stemming, lemmatization, etc.) and vectorization methods (binary, count, TF-IDF).
- **Model Training**: Trains different machine learning models (Logistic Regression, SVM, Random Forest) on the processed datasets and evaluates their performance. The best hyperparameters for each model are determined using GridSearchCV.
- **Sentiment Prediction**: Provides functionality to predict the sentiment of new text inputs using the trained models.

### 2. `utils/text_preprocessing.py`
This module contains functions for preprocessing text data. The key functionalities include:
- **Tokenization**: Splits text into individual words.
- **Stemming**: Reduces words to their base or root form using the Porter Stemmer.
- **Lemmatization**: Converts words to their base form based on their part of speech using the WordNet lemmatizer.
- **Spelling Correction**: Corrects misspelled words using the SymSpell algorithm.
- **Remove Punctuation**: Eliminates punctuation from the text to clean the data.


### 3. `utils/text_vectorization.py`
This module provides functions for converting text data into numerical vectors suitable for machine learning models. The key functionalities include:
- **Binary Vectorization**: Converts text into binary vectors indicating the presence or absence of words.
- **Count Vectorization**: Converts text into count vectors representing the frequency of words.
- **TF-IDF Vectorization**: Converts text into TF-IDF vectors that reflect the importance of words in the context of the entire dataset.


### How to use
1. Ensure you have the required libraries installed. the requirements.txt file can be used to see the required libraries.
2. Run the Jupyter Notebook `sentiment_analysis.ipynb` to execute the sentiment analysis workflow
