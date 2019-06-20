import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import sys
import pandas as pd
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target.

    Arguments:
    database_filepath -- path to location of database

    Returns:
    X -- features in the form of a message
    y -- multiple target variables
    category_names -- list of all target variable names
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message', con=engine)

    # assign values
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Apply case normalization, lemmatize and tokenize text.

    Arguments:
    text -- text in source format

    Returns:
    clean_tokens -- cleaned tokenized list
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize, remove stop words and whitespace
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Model that handles multi-output classification and use grid search.

    Arguments:
    None

    Returns:
    cv -- multi-output classification model
    """
    # define pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # set parameters for grid search
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 100, 400),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [1],
        'clf__estimator__min_samples_split': [2],
        'clf__estimator__random_state':[42],
        'clf__estimator__n_jobs': [4],
        'clf__estimator__verbose': [0]
    }

    # perform grid search
    cv = GridSearchCV(pipeline,
                      param_grid=parameters,
                      cv=2,
                      verbose=1)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    Predict values and key metrics are presented for each category.

    Arguments:
    model -- classification model used for prediction
    X_test -- features in the form of a message for test set
    y_test -- multiple target variables for test set
    category_names -- list of all target variable names

    Returns:
    None
    """
    # predict values
    y_pred = model.predict(X_test)

    # display results for each category
    for i in range(0, len(category_names)):
        print('-'*80)
        print('Result for: {}'.format(category_names[i]))
        df = pd.DataFrame(classification_report(y_test.iloc[:,i],
                                    y_pred.transpose()[i],
                                    output_dict=True))
        print(df)

    # display best model parameters
    print("\nBest Parameters:", model.best_params_)


def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Arguments:
    model -- classification model to be saved
    model_filepath -- file path to save model as

    Returns:
    None
    """
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('\nLoading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('\nBuilding model...')
        model = build_model()

        print('\nTraining model...')
        model.fit(X_train, y_train)

        print('\nEvaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('\nSaving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('\nTrained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
