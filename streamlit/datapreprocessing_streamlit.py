import streamlit as st
import os

import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

########################################### Class Datapreprocessing ##########################################

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
        return X_train, y_train
    
    @staticmethod
    def get_weights(Y_train):
        class_weights = dict(zip(np.unique(Y_train), (1 / np.bincount(Y_train))))
        return class_weights


########################################### streamlit select data #############################################

st.title('Data Preprocessing for Cancer Classification')

def file_selector(folder_path = './Tan_data-2'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A File', filenames)
        return os.path.join(folder_path, selected_filename)

filename = file_selector()

#select data
st.info('You Selected {}'.format(filename))


########################################## datapreprocessing function #########################################

def automate_data_processing(filename):

    #getting data
    data = DataPreprocessing.get_df(data_path = filename)

    #getting features and labels
    features, labels = DataPreprocessing.get_features_labels(data)

    #one-hot encode y
    labels = DataPreprocessing.get_encoder(labels).transform(labels)

    #get original train and tests
    X_train, X_test, Y_train, Y_test = DataPreprocessing.get_train_test_sets(features, labels)


    if st.checkbox('Show Dataset'):
        number = st.number_input('Number of Rows', 5, 10)
        st.dataframe(data.head(number))

        #get weights 
        if st.checkbox('Show Weights of Binary Class'):
            #weight_display = weight_table.rename(columns = {'':'Class', '0':'Weights'})
            st.table(pd.DataFrame(list(DataPreprocessing.get_weights(Y_train).items()), columns=['Class', 'Weight']))

            #get train and tests after SMOTE (only change for train)
            if st.checkbox('Balance Train Dataset'):
                X_train, Y_train = DataPreprocessing.get_smote(X_train, Y_train)
                st.table(pd.DataFrame(list(DataPreprocessing.get_weights(Y_train).items()), columns=['Class', 'Weight']))
    return X_train, Y_train, X_test, Y_test

########################################### testing function ###################################################

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = automate_data_processing(filename)