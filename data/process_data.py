import sys
# import libraries
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories data from csv filepaths.
    Returns one dataframe of the two datasets merged together on the id column 
    
    input:
    two filepath strings for the messages and categories csv files
    
    return:
    Dataframe and the two csv files merged on the id column
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id', how = 'left')
    
    return df


def clean_data(df):
    """
    Takes merged df from load_data function as input. 
    Returns cleaned dataframe prepared for NLP classification model
    
    input:
    merged df from load_data function
    
    return:
    Cleaned dataframe for NLP classification model
    """
    categories = df['categories'].str.split(';', expand=True)
    category_colnames = list(categories.iloc[0].apply(lambda x: x[:-2]).values)
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(float)
    
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1, join='inner', sort=False)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    Given a dataframe and database_filename uses SQLAlchemy and sqlite3 to create a SQL table of the dataframe
    
    input:
    dataframe, text string 
    
    return:
    None
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql(database_filename, engine, if_exists = 'replace', index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()