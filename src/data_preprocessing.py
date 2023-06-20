import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import math 
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data_path = '/Users/christinaxu/Documents/genomic-cancer-classification/data/Colon.txt'

class DataPrepocessing:
    X_train = None
    y_train = None 
    X_test = None 
    y_test = None

    @staticmethod
    def get_df(data_path):
        df = pd.read_csv(data_path,  sep=',', header=None).T
        return df 
    
    @staticmethod
    def get_features_labels(df):
        features = df.iloc[:, 1:].astype(float)
        labels = df.iloc[:, 0]
        return features, labels

    @staticmethod
    def get_train_test(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def get_encoder(labels):
        le = LabelEncoder()
        encoder = le.fit(labels)
        return encoder

    @staticmethod
    def scale_to_rgb(sample):
        norm_sample = sample/255
        rounded_up_sample = (np.ceil(norm_sample)).astype(int)
        return rounded_up_sample
    
    @staticmethod
    def features_to_imgs(features):
        if (math.sqrt(features.shape[1])).is_integer() == False:
            dim = math.ceil(math.sqrt(features.shape[1]))
        else: 
            dim = math.sqrt(features.shape[1])
        
        imgs = []
        zeros = [0] * (dim ** 2 - features.shape[1])

        for row in range(len(features)):
            # reshapes sample to image
            sample = features.iloc[row,:].values.tolist()
            sample += zeros 
            sample = np.reshape(a=sample, newshape=(dim, dim))
            sample = DataPrepocessing.scale_to_rgb(sample)
            imgs.append(sample)
            
        return imgs
    
    @staticmethod
    def display_imgs(imgs,labels):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(imgs[i+random.randint(0, 62)].astype("uint8"))
            plt.title(labels[i+random.randint(0, len(imgs))])
            plt.axis("off")

    
    # def automate_data_preprocessing(data_path):
    #     print('loading data...')
    #     DataPrepocessing.df = DataPrepocessing.get_df(data_path=data_path)
    #     # seperate df into X, y 
    #     print('getting features and labels')
    #     DataPrepocessing.features, DataPrepocessing.labels = DataPrepocessing.get_features_labels(DataPrepocessing.df)
    #     # transform X into imgs
    #     print('transforming features into images')
    #     DataPrepocessing.imgs = DataPrepocessing.features_to_imgs(DataPrepocessing.features)
    #     # display imgs
    #     DataPrepocessing.display_imgs(DataPrepocessing.imgs, DataPrepocessing.labels)
    #     # transform imgs in array
    #     DataPrepocessing.X = np.array(DataPrepocessing.imgs)
    #     # one-hot encode y 
    #     DataPrepocessing.y_encoded = DataPrepocessing.get_encoder(DataPrepocessing.labels).transform(DataPrepocessing.labels)
    #     DataPrepocessing.X_train, DataPrepocessing.X_test,  DataPrepocessing.y_train, DataPrepocessing.y_test = DataPrepocessing.get_train_test(DataPrepocessing.X, DataPrepocessing.y_encoded)




