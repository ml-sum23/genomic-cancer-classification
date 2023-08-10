import streamlit as st
import sys 
sys.path.append("../src")
from data_preprocessing import * 
import time

def main():
    st.title('Cancer Genomics Classification')
    cancer_type = st.selectbox('Select Cancer Type', ['Colon', 'DLBCL', 'GCM', 'Liver'])
    # st.write(f"{cancer_type} selected.")
    data_path = f"/Users/christinaxu/Documents/genomic-cancer-classification/data/{cancer_type}.txt"
    

    if cancer_type: 
        df = DataPrepocessing.get_df(data_path)
        st.write(f"Displaying the first 5 rows of {cancer_type} data...")
        st.dataframe(df.head())
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
                # y_train = y_train.ravel()
                # X_train = X_train.reshape(-1, X_train.shape[1], X_train.shape[2], 1)
                print(f"X train dims: {X_train.shape}\ny train dims: {y_train.shape}")
            groups = []
            # plot the first 9 imgs and their labels
            for i in range(0, len(imgs[:9]), 3):
                groups.append(imgs[i:i+3])
            for group in groups:
                cols = st.columns(3)
                for i, img in enumerate(group):
                    cols[i].image(img)

            if 'X_train' not in st.session_state:
                st.session_state['X_train'] = X_train
            if 'X_test' not in st.session_state:
                st.session_state['X_test'] = X_test
            if 'y_train' not in st.session_state:
                st.session_state['y_train'] = y_train
            if 'y_test' not in st.session_state:
                st.session_state['y_test'] = y_test
            
            st.success("Data preparation successful. Please proceed to training the model ✅")
        
if __name__=='__main__':
    main()