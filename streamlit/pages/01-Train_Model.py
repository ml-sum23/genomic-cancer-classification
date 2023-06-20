import matplotlib.pyplot as plt
import streamlit as st
import sys
sys.path.append("../src")
from cnn_model import *


def main():
    st.title('🏋🏻‍♂️ Train Model')
    cancer_type = st.session_state["cancer_type"]
    model_path = f'../models/{cancer_type}.keras'

    # with st.spinner(text="Loading training data..."):
    X_train, y_train = st.session_state['X_train'], st.session_state['y_train']
    success = st.success("""
                         Training data successfully loaded ✅. Please proceed to selecting the number of epochs to train the model for 👇🏻
                        """)
    if success:
        epochs = st.select_slider('Epochs', range(0, 100, 1))

        if epochs != 0:
            train_model = st.button("Train Model")
            if train_model:
                st.write(f"Starting training with {epochs} epochs...")
                print("Loading model...")
                print(X_train[0].shape)
                
                model = CNNModel(input_shape=X_train[0].shape,
                                n_hidden_nodes=64, 
                                kernel_size=4)
                model.compile(learning_rate=0.001)
                print("Training in process...")
                epochs = int(epochs)
                history = model.fit(X=X_train,
                                        y=y_train, 
                                        epochs=epochs)
                        
                fig1, ax1 = plt.subplots()
                plt.plot(history.history['accuracy'], color='blue')
                        # plt.plot(history.history['val_accuracy'])
                plt.title('Model Training Accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots()
                plt.plot(history.history['loss'], color='orange')
                plt.title('Model Training Loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                    
                st.pyplot(fig2)

                model.save_model(model_path)
                        
                st.success('Training is complete ✅. Please proceed to model prediction 👉🏻')
                
if __name__=='__main__':
    main()