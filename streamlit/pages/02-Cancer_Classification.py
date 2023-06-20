import streamlit as st
import sys
sys.path.append("../src")
import warnings

from cnn_model import * 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ignore warning message from st.pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("👩🏻‍🔬 Cancer Classification")
    cancer_type = st.session_state["cancer_type"]
    model_path = f'../models/{cancer_type}.keras'
    X_test, y_test = st.session_state['X_test'], st.session_state['y_test']
    model = CNNModel.load_model(model_path)
    success = st.success('Trained model and test data sucessfully loaded ✅')
    
    if success:
        st.subheader("Confusion Matrix")
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

if __name__=='__main__':
    main()
