from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score

from nnmodel_streamlit import *


st.title('Evaluation for Multiple Perceptrons Neural Network Classification Model')

def file_selector(folder_path = './Tan_data-2'):
        filenames = os.listdir(folder_path)
        #selected_filename = st.selectbox('Select A File', filenames)
        selected_filename = st.selectbox('Select A File', filenames, key="file_selector")
        return os.path.join(folder_path, selected_filename)


filename = file_selector()
#file_path = os.path.join('./Tan_data-2', selected_filename)

#select data
st.info('You Selected {}'.format(filename))

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

def evaluation(model):
    st.subheader('Confusion Matrix')
    y_pred = model.predict(X_test)
    y_pred = [np.round(x) for x in y_pred]
    confusion_mtx = confusion_matrix(y_test, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_mtx)
    cmd.plot()  
    st.pyplot()

    sen, spec, bal_acc = CNNModel.get_evaluation_metrics(y=y_test, y_pred=y_pred)
    col1, col2, col3 = st.columns(3)
    col1.metric("Sensitivity", f"{sen}")
    col2.metric("Specificity", f"{spec}")
    col3.metric("Balanced Accuracy", f"{bal_acc}")

########################################### testing function ###################################################

if __name__ == '__main__':
    
    X_train, Y_train, X_test, Y_test = automate_data_processing(filename)
    model = NNModel.load_model(filename)
    model = nnmodel_fit(X_train, Y_train, X_test, Y_test, hidden_layers=hidden_layers, learning_rate=learning_rate, epochs=epochs, patience=patience)
    evaluation(model)


    




