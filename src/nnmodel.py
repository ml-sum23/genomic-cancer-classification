

import streamlit as st

import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LeakyReLU
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

st.sidebar.header('User Input Parameters')

hidden_layers = st.sidebar.slider('Hidden Layers', 2, 12, 1)
learning_rate = st.sidebar.slider('Learning Rate', 0.000001, 1.0, 0.00001)
epochs = st.sidebar.slider('Epochs', 50, 500, 50)
patience = st.sidebar.slider('Patience', 0, 300, 50)
    
def user_input_features():
    data = {'hidden_layers' : hidden_layers,
            'learning_rate' : learning_rate,
            'epochs' : epochs,
            'patience' : patience}
    input = pd.DataFrame(data, index=[0])
    return input

input_df = user_input_features()

st.subheader('User Input Parameters')
st.write(input_df)


class NNModel:
    def __init__(self,
                 hidden_layers = None):
        seed_value = 42
        tf.random.set_seed(seed_value)
        #num_hidden_layers = hidden_layers[0]
        self.model = Sequential()
        x = 0
        #for x in range(num_hidden_layers+4,4, -1):
        for x in range(hidden_layers+4,4, -1):
            #self.model.add(Dense(2**(num_hidden_layers)))
            self.model.add(Dense(2**(hidden_layers)))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation='sigmoid'))
    
    def compile(self, learning_rate):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        
    def fit(self, 
            X_train, 
            Y_train, 
            X_test,
            Y_test,
            epochs=None,
            patience = None,
            class_weights = None):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
        history = self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=epochs,  verbose=0, callbacks=[es], class_weight=class_weights)
        return history
    
    def predict(self, X):
        y = self.model.predict(X)
        y = [np.round(x) for x in y]
        return y 
    
    def get_evaluation_metrics(self, X, y):
        y_pred = self.predict(X)
        y_pred_classes = [np.round(x) for x in y_pred]
        tn, fp, fn, tp = confusion_matrix(y, y_pred_classes).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp/(tp+fp)
        print(f'Sensitivity: {sensitivity}')
        bal_acc = (specificity + sensitivity) / 2
        print(f'Balanced accuracy: {bal_acc}')
        f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)
        print("F1 score:", f1_score)
        return specificity, sensitivity, bal_acc, f1_score
        
    def save_model(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return models.load_model(model_path)
    
def nnmodel_fit(X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate=None, epochs=None, patience=None, class_weights=None):
    # Create an instance of NNModel
    model = NNModel(hidden_layers=hidden_layers)
    
    # Compile the model
    model.compile(learning_rate=learning_rate)
    
    # Fit the model to the data
    history = model.fit(X_train, Y_train, X_test, Y_test, epochs=epochs, patience=patience, class_weights=class_weights)
    
    # Evaluate the model
    specificity, sensitivity, bal_acc, f1_score = model.get_evaluation_metrics(X_test, Y_test)
    
    # Return the evaluation metrics and training history
    return specificity, sensitivity, bal_acc, f1_score, history

X_train, X_test, Y_train, Y_test = automate_data_processing(data_path)
nnmodel_fit(X_train, Y_train, X_test, Y_test, hidden_layers=(5,), learning_rate=0.0001, epochs=300, patience=50)