import streamlit as st
import sys
sys.path.append("../src")
from cnn_model import * 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    st.spinner('Loading trained model and test data...')
    X_test, y_test = st.session_state['X_test'], st.session_state['y_test']
    success = st.success('Trained model and test data loaded ✅')
    
    if success:
        st.subheader("Confusion Matrix")
        y_pred_classes = CNNModel.predict(X=X_test)
        confusion_mtx = confusion_matrix(y_test,y_pred_classes)
        cmd = ConfusionMatrixDisplay(confusion_mtx)
        cmd.plot()  
        st.pyplot()
        
        spec, sen, bal_acc = CNNModel.get_evaluation_metrics(X_test, y_test)

        st.metric("Specificity", f"{spec}")
        st.metric("Sensitivity", f"{sen}")
        st.metric("Balanced Accuracy", f"{bal_acc}")

if __name__=='__main__':
    main()
