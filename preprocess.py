import sys
import pandas
import numpy as np
from pandas.api.types import is_numeric_dtype

def removeUnknownRecords(df):
    for attr in df.columns:
        df.drop(df[df[attr].astype(str).str.strip() == '?'].index, inplace=True)
    return df

def removeContinuousAttributes(df):
    'Remove all continuous attributes from a csv'
    # Strips whitespace from attribute names
    df.rename(columns=lambda x: x.strip(), inplace=True)
    df.drop(columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], inplace=True)
    return df

def oneHotEncodeMultiDomain(df):
    for attr in df.columns:
        print(attr + ': ' + df[attr].dtype.name)
        if df[attr].dtype.name == 'category':
            oneHot = pandas.get_dummies(df[attr], prefix=attr)
            df.drop(columns=attr, inplace=True)
            # Fix for '?' values somehow being used even though it's not in any records
            for a in oneHot.columns:
                if '?' in str(a) and oneHot[a].nunique() == 1:
                    oneHot.drop(columns=a, inplace=True)
            df = pandas.concat([df, oneHot], axis=1)
    return df

def numericToBinary(df):
    for attr in df.columns:
        print(is_numeric_dtype(df[attr]))
        if is_numeric_dtype(df[attr]):
            mean = round(df[attr].mean(), 2)
            print(str(attr) + ' mean: ' + str(mean))
            df[attr] = (df[attr] < mean).apply(lambda x: '<' + str(mean) if x else '>=' + str(mean)).astype(str)
    return df

def lastNRecords(df, n):
    print(df.tail(n))
    return df.tail(n)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        modifiedFilename = 'modified.' + filename
        if len(sys.argv) > 2:
            modifiedFilename = sys.argv[2]
        dtypes = {
            'age': np.int32,
            'workclass': 'category',
            'fnlwgt': np.int64,
            'education': 'category',
            'education-num': np.int8,
            'martial-status': 'category',
            'occupation': 'category',
            'relationship': 'category',
            'race': 'category',
            'capital-gain': np.int32,
            'capital-loss': np.int32,
            'hours-per-week': np.int8,
            'native-country': 'category',
        }
        df = pandas.read_csv(filename, sep=', ', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'], dtype=dtypes)
        running = True
        while running:
            choices = '''
                        1) Remove unknown records
                        2) Remove Continuous Attributes
                        3) Save output to CSV and quit
                        4) Numeric to binary
                        5) One Hot Encode multi-domain
                        6) Choose last 10 records (For kNN on number 2)
                        a) Shortcut for number 1 (remove unknown, remove continuous attributes)
                        b) Shortcut for number 2 (remove unknown, numeric to binary, one hot encode)
                        '''
            promptMsg = 'Please enter the choice for what you want to do to the data:\n' + choices
            response = input(promptMsg)
            if response == '1':
                df = removeUnknownRecords(df)
            elif response == '2':
                df = removeContinuousAttributes(df)
            elif response == '3':
                print(df)
                df.to_csv(modifiedFilename, index=False)
                running = False
                print('Saving to ' + str(modifiedFilename) + ' and quitting...')
            elif response == '4':
                df = numericToBinary(df)
            elif response == '5':
                df = oneHotEncodeMultiDomain(df)
            elif response == '6':
                df = lastNRecords(df, 10)
            elif response == 'a':
                df = removeUnknownRecords(df)
                df = removeContinuousAttributes(df)
            elif response == 'b':
                df = removeUnknownRecords(df)
                df = numericToBinary(df)
                df = oneHotEncodeMultiDomain(df)
        

    