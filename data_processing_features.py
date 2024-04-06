# import libraries
# import sqlite3
import sqlite3

#import pandas
import pandas as pd


# get data from the database
def get_data(table):
    # create sqlite connection
    con = sqlite3.connect('/Users/puneetkucheria/projects/data_science_course/capstone projects/Database.db')
    # print(pd.read_sql_query("SELECT * FROM sqlite_master", con))
    df = pd.read_sql_query("SELECT * FROM " + table + "", con) # Fraud_detection
    print("data imported")
    return df

# convert columns to numarical
def convert_to_numeric(df, num_col):
    #converting object type columns in to numeric.
    for col in num_col:
        df[col] = pd.to_numeric(df[col])
    print("converted to numeric")
    return df


# data transformation
def data_transformation(df):
    # handling null values
    df['type'] = df['type'].replace('', 'NOTYPE')

    condition = df['type'].isin(['DEBIT', 'CASH_OUT', 'TRANSFER', 'PAYMENT', 'NOTYPE'])
    df.loc[condition, 'newbalanceOrig'] = df.loc[condition, 'oldbalanceOrg'] - df.loc[condition, 'amount']

    condition = (df['type'] == 'CASH_IN')
    df.loc[condition, 'newbalanceOrig'] = df.loc[condition, 'oldbalanceOrg'] + df.loc[condition, 'amount']

    condition = (df['oldbalanceDest'].isna())
    df.loc[condition,'oldbalanceDest'] = 0.0

    df = df.drop('nameOrig', axis=1)
    df = df.drop('nameDest', axis=1)   

    # creating dummy variables for categorical data Type
    df=pd.get_dummies(df, columns=['type'], drop_first=True)

    print("data transformed")
    return df


