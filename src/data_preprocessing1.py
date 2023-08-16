
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data_path = '/Users/nguyencaogiakhanh/Desktop/MLSummer23/Tan_data-2/Colon.txt'

class DataPreprocessing:
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

    @staticmethod
    def get_df(data_path):
        df= pd.read_csv(data_path, sep = ',', header = None).T
        return df
    
    @staticmethod
    def get_features_labels(df):
        features = df.iloc[:,1:].astype(float)
        labels = df.iloc[:,0]
        return features, labels
    
    @staticmethod
    def get_train_test_sets(X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, shuffle=True, random_state=42)
        print(X_train.shape, Y_train.shape)
        print(X_test.shape, Y_test.shape)
        return X_train, X_test, Y_train, Y_test
    
    @staticmethod
    def get_encoder(labels):
        le = preprocessing.LabelEncoder()
        encoder = le.fit(labels)
        return encoder
    
    @staticmethod
    def get_smote(X_train,Y_train):
        smote = SMOTE(sampling_strategy='auto') 
        X_train, y_train= smote.fit_resample(X_train, Y_train)
        print(X_train.shape, y_train.shape)
        return X_train, y_train
    
    @staticmethod
    def get_weights(Y_train):
        class_weights = dict(zip(np.unique(Y_train), (1 / np.bincount(Y_train))))
        print(class_weights)
        return class_weights
    
    
def automate_data_processing(data_path):

    #getting data
    print('loading data...')
    data = DataPreprocessing.get_df(data_path = data_path)
    print('getting features and labels')

    #getting features and labels
    features, labels = DataPreprocessing.get_features_labels(data)
    
    #one-hot encode y
    labels = DataPreprocessing.get_encoder(labels).transform(labels)

    #get original train and tests
    X_train, X_test, Y_train, Y_test = DataPreprocessing.get_train_test_sets(features, labels)

    #get train and tests after SMOTE (only change for train)
    DataPreprocessing.X_train, DataPreprocessing.Y_train = DataPreprocessing.get_smote(X_train, Y_train)

    #get weights 
    DataPreprocessing.get_weights(Y_train)
    return X_train, Y_train, X_test, Y_test

