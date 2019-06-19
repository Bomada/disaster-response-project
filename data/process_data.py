import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from csv files into a pandas dataframe.

    Arguments:
    messages_filepath -- path to location of messages csv file
    categories_filepath -- path to location of categories csv files

    Returns:
    df -- pandas dataframe with messages and categories
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, how='outer', on='id')

    return df


def clean_data(df):
    """
    Clean data in a pandas dataframe.

    Clean data in a pandas dataframe by renaming category columns, convert
    category values to 0 or 1 and drop duplicates.

    Arguments:
    df -- pandas dataframe with messages and categories in source format

    Returns:
    df -- cleaned pandas dataframe with messages and categories
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # use first row to extract a list of new column names for categories
    row = categories.iloc[0,]
    category_colnames = []
    for category in row:
        category_colnames.append(category[0:-2])

    # rename the columns of categories
    categories.columns = category_colnames

    # convert category values to 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int)

    # replace categories column in df with new category columns
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save data from a pandas dataframe into SQLite database.

    Arguments:
    df -- pandas dataframe with messages and categories
    database_filename -- path to location of database file

    Returns:
    None
    """
    # save dataset into database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message', engine, index=False, if_exists='replace')


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
