import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import sqlite3
from sqlalchemy import create_engine
import joblib 


def load_data(database_filepath):
    """
    Reads cleaned data from SQL that was stored into DisasterReponse.db when running process_data.py
    Returns the data as a DataFrame 
    
    input:
    datebase_filepath : str 
    
    return:
    dataframe X for X variables, dataframe Y for y or target variables, and a list of the category names
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table(database_filepath, engine)
    
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'], axis=1)
    category_names_list = list(Y.columns)
    
    return X, Y, category_names_list


def tokenize(text):
    """
    Tokenizes the messages so that they can be used as the input for the NLP classification model
    
    input:
    text string 
    
    return:
    list of tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def build_model():
    """
    Creates a GridSearchCV object using NLP feature engineeing pipeline 
    and a multi output random forest classifier
    
    input:
    None 
    
    return:
    Gridsearch CV object that has been instantiated using our NLP feature engineering pipeline and a multi output 
    Random Forest Classifier
    """
    #create the pipeline to vectorize, tfidf transform and train a multioutput random forest classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #use non computationally intensive parameter options to make training time feasible 
    parameters = {
        'clf__estimator__n_estimators': [5, 10],
        'clf__estimator__min_samples_split': [2]
    }

    # Create the grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Generates model predictions and prints the f1 score, precision, and recall for each category
    
    input:
    trained / fit model, X_test dataframe, Y_test dataframe, and list of category_names
    
    return:
    None
    """
    y_pred = model.predict(X_test)
    labels = category_names
    # Iterate through each output category
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[category], y_pred[:,i]))
        print("---------------------------------------------")
    accuracy = (y_pred == Y_test.values).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    pass


def save_model(model, model_filepath):
    """
    Saves the input model to a given model filepath
    
    input:
    model object and model file_path string
    
    return:
    None
    """
    # Train your model and obtain the trained model object
    model = model

    # Specify the filepath where you want to save the model
    model_filepath = model_filepath

    # Save the model as a pickle file
    joblib.dump(model, model_filepath)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()