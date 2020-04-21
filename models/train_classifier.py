import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle
import warnings

def load_data(database_filepath):
    """
    Function to load data from database and split data to X,Y data set for ML.
    Args: database_filepath.
    Return:X, Y data sets and category_names of Y dataset. 
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponseTable', engine)
    category_names = df.columns[4:]
        
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize text message.
    Args: text.
    Return: A list of tokens from text. 
    """
    raw_toks = word_tokenize(text)
    lem = WordNetLemmatizer()
    tokens = []
    for tok in raw_toks:
        tokens.append(lem.lemmatize(tok).strip().lower())        
    return tokens


def build_model():
    """
    Function to build a ML model.
    Args: None.
    Return: A ML pipeline. 
    """ 
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()),
                     ('clf', MultiOutputClassifier(AdaBoostClassifier(base_estimator=RandomForestClassifier(),
                        n_estimators=100, learning_rate=0.9,
                        random_state=None ))) ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Function to evaluate a ML model.
    Args: ML model, X_test dataframe, Y_test dataframe, list of category_names.
    Return: A ML pipeline. 
    """ 
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print('............................')
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Function to save ML model by pickle.
    """ 
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
       
        # train the model and catch future warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
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