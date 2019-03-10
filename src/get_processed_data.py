import os
import pandas as pd
import numpy as np


def read_data():
    # raw data
    train_df = pd.read_csv('../rawdata/train.csv', index_col='PassengerId')
    test_df = pd.read_csv('../rawdata/test.csv', index_col='PassengerId')
    test_df['Survived'] = -111
    df = pd.concat((train_df, test_df), axis=0)
    return df

def process_data(df):
    return(df
          # create title attribute
           .assign(Title = lambda x : x.Name.map(get_title))
           # missing values
           .pipe(fill_missing_values)
           # create age state either adult or child
           .assign(AgeState = lambda x : np.where(x.Age >=18, 'Adult', 'Child'))
           #family size
           .assign(FamilySize = lambda x : (x.Parch + x.SibSp + 1))
           # create deck feature
           .assign(Cabin = lambda x : np.where(x.Cabin == 'T', np.nan, x.Cabin))
           .assign(Deck = lambda x : x.Cabin.map(get_deck))
           # feature encoding
           .assign(IsMale = lambda x : np.where(x.Sex == 'male', 1, 0))
           .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Embarked', 'AgeState'])
           #drop missing columns
           .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
           
           #re order columns
           .pipe(reorder_columns)
          )

def get_title(name):
    title_group = {
        'mr' : 'Mr',
        'miss' : 'Miss',
        'mrs' : 'Mrs',
        'master' : 'Master',
        'dr' : 'Dr',
        'rev' : 'Sir',
        'major' : 'Officer',
        'mlle' : 'Miss',
        'col' : 'Officer',
        'jonkheer'  : 'Sir',
        'mme' : 'Mrs',
        'don' : 'Sir',
        'capt' : 'Officer',
        'lady' : 'Miss',
        'sir' : 'Sir',
        'the countess' : 'Miss',
        'ms' : 'Miss',
        'dona' : 'Mrs'
    }
    name_with_title = name.split(',')[1]
    title = name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def fill_missing_values(df):
    #embarked
    df.Embarked.fillna('C', inplace=True)
    #age
    title_age_median = df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    #fare
    median_fare = df[(df.Pclass == 3) & (df.Embarked == 'S')]['Fare'].median()
    df.Fare.fillna(median_fare, inplace=True)
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def write_data(df):
    
    #training data
    df[df.Survived != -111].to_csv('../processed/train.csv')
    #test data
    columns = [column for column in df.columns if column != 'Survived']
    df[df.Survived == -111].to_csv('../processed/test.csv')
    return df
    

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    df = write_data(df)
    
    
    
    
