import streamlit as st
import sys 
sys.path.append("../src")
from data_preprocessing import * 
import time

# ignore warning message from st.pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title('Data Preprocessing')

    cancer_type = st.selectbox('Select Cancer Type', ['Colon', 'DLBCL', 'GCM', 'Leukemia', 'Lung', 'Prostate1', 'Prostate2', 'Prostate3'])
    # st.write(f"{cancer_type} selected.")
    data_path = f"../data/{cancer_type}.txt"
    
    if cancer_type not in st.session_state:
        st.session_state["cancer_type"] = cancer_type
    
    # option = st.radio(
    #     "Select model type", 
    #     key="visibility", 
    #     options=['Neural Network (NN)', 'Convolutional Neural Network']
    # )
    
    if cancer_type: 
        df = DataPrepocessing.get_df(data_path)
        # checkbox to display first 5 rows of data
        display_data = st.checkbox(f'Display first 5 rows of {cancer_type} data')
        if display_data: 
            st.dataframe(df.head())
        # prepare data button 
        prep_data = st.button("Prepare Data")
        if prep_data:
            with st.spinner(text="Data preprocessing in progress..."):
                time.sleep(10)
                features, labels = DataPrepocessing.get_features_labels(df)
                imgs = DataPrepocessing.features_to_imgs(features)
                # display imgs 
                X = np.array(imgs)
                y_encoded = DataPrepocessing.get_encoder(labels).transform(labels)
                
                X_train, X_test, y_train, y_test = DataPrepocessing.get_train_test(X, y_encoded)
                print(f"X train dims: {X_train.shape}\ny train dims: {y_train.shape}")

            # plot the first 9 imgs and their labels
            plt.figure(figsize=(10, 10))
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(imgs[i].astype("uint8"))
                plt.title(labels[i])
                plt.axis("off")
            st.pyplot()

            if 'X_train' not in st.session_state:
                st.session_state['X_train'] = X_train
            if 'X_test' not in st.session_state:
                st.session_state['X_test'] = X_test
            if 'y_train' not in st.session_state:
                st.session_state['y_train'] = y_train
            if 'y_test' not in st.session_state:
                st.session_state['y_test'] = y_test
            
            st.success("Data preparation successful ✅. Please proceed to training the model 👉🏻")
        
if __name__=='__main__':
    main()