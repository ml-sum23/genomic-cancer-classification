import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class CNNModel:
    def __init__(self,
                 input_shape=None,
                 n_hidden_nodes=None,
                 kernel_size=None
                 ):

        seed_value = 42
        tf.random.set_seed(seed_value)
        self.model = Sequential()
        self.model.add(Conv2D(n_hidden_nodes, kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding="same", input_shape=(input_shape[0], input_shape[1], 1)))
        self.model.add(Conv2D(n_hidden_nodes*2, kernel_size=(int(kernel_size/2), int(kernel_size/2)), strides=(1, 1), padding="same", activation='relu'))
        self.model.add(Conv2D(n_hidden_nodes*4, kernel_size=(int(kernel_size/4), int(kernel_size/4)), strides=(1, 1), padding="same", activation='relu'))

        self.model.add(MaxPooling2D((2, 2), strides=2))
        self.model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
        self.model.add(Dropout(rate=0.25))
        self.model.add(Flatten())

        self.model.add(Dense(n_hidden_nodes*2, activation='relu'))
        self.model.add(Dense(n_hidden_nodes*2, activation='relu'))
        self.model.add(Dense(n_hidden_nodes/2,activation='relu'))
        self.model.add(Dense(1,activation='sigmoid'))


    def compile(self, learning_rate):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, 
            X, 
            y, 
            epochs=None):
        history = self.model.fit(X, y, epochs=epochs)
        return history
    
    def save(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return keras.models.load_model(model_path)
    
    def predict(self, X):
        y = self.model.predict(X)
        y = [np.round(x) for x in y]
        print(y)
        return y 
    
    @staticmethod
    def get_evaluation_metrics(y, y_pred):
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
        sensitivity = tp/(tp+fn) 
        specificity = tn/(tn+fp)
        bal_acc = (specificity + sensitivity) / 2
        return  sensitivity, specificity, bal_acc

    def save_model(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return models.load_model(model_path)





