
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

from datapreprocessing_streamlit import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


 ########################################### Class NNModel ##########################################

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
        matrix = confusion_matrix(y, y_pred_classes)
        tn, fp, fn, tp = confusion_matrix(y, y_pred_classes).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp/(tp+fp)
        print(f'Sensitivity: {sensitivity}')
        bal_acc = (specificity + sensitivity) / 2
        print(f'Balanced accuracy: {bal_acc}')
        f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)
        print("F1 score:", f1_score)
        return specificity, sensitivity, bal_acc
    
    
    def save_model(self, model_path):
        self.model.save(model_path)

    @staticmethod
    def load_model(model_path):
        return models.load_model(model_path)
    

########################################### NNModel Func ##########################################

def nnmodel_fit(X_train, Y_train, X_test, Y_test, hidden_layers, learning_rate=None, epochs=None, patience=None, class_weights=None):
    # Create an instance of NNModel
    model = NNModel(hidden_layers=hidden_layers)
    
    # Compile the model
    model.compile(learning_rate=learning_rate)
    
    # Fit the model to the data
    history = model.fit(X_train, Y_train, X_test, Y_test, epochs=epochs, patience=patience, class_weights=class_weights)
    # Plot the accuracy
    # Streamlit app title
    st.title('Model Accuracy')

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot training accuracy and validation accuracy
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])

    # Set title, labels, and legend
    ax.set_title('Model Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(['Train', 'Validation'], loc='upper left')

    st.pyplot(fig)
    # plot training history
    st.title('Model Loss')

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot training loss and validation loss
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])

    # Set title, labels, and legend
    ax.set_title('Model Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(['Train', 'Validation'], loc='upper right')

    st.pyplot(fig)

    #model.save_model(filename)
    st.subheader("Confusion Matrix")
    y_pred = model.predict(X_test)
    y_pred = [np.round(x) for x in y_pred]
    confusion_mtx = confusion_matrix(Y_test, y_pred)
    fig, ax = plt.subplots()
    cmd = ConfusionMatrixDisplay(confusion_mtx)
    cmd.plot(ax=ax)
    ax.set_title('Confusion Matrix') 
    st.pyplot(fig)
        
    return history

########################################### streamlit select data #############################################

st.title('Multiple Perceptrons Neural Network Model for Cancer Classification')

def file_selector(folder_path = './Tan_data-2'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select A File', filenames)
        return os.path.join(folder_path, selected_filename)

filename = file_selector()

#select data
st.info('You Selected {}'.format(filename))



########################################### streamlit user inputs ##############################################

def user_input_features():
        if st.checkbox('Change Parameters'):
            st.sidebar.header('User Input Parameters')

            hidden_layers = st.sidebar.slider('Hidden Layers', 2, 12, 1)
            learning_rate = st.sidebar.slider('Learning Rate', 0.0001, 0.005, 0.0001)
            epochs = st.sidebar.slider('Epochs', 50, 500, 50)
            patience = st.sidebar.slider('Patience', 0, 300, 50)
        else:
            hidden_layers = 5
            learning_rate = 0.0001
            epochs = 300
            patience = 50    
        return hidden_layers, learning_rate, epochs, patience
    
hidden_layers, learning_rate, epochs, patience = user_input_features()

data = {'hidden_layers' : hidden_layers,
        'learning_rate' : learning_rate,
        'epochs' : epochs,
        'patience' : patience}

input = pd.DataFrame(data, index=[0])
st.dataframe(input)

########################################### testing function ###################################################

if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = automate_data_processing(filename)
    nnmodel_fit(X_train, Y_train, X_test, Y_test, hidden_layers=hidden_layers, learning_rate=learning_rate, epochs=epochs, patience=patience)
    

