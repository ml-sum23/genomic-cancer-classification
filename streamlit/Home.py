import streamlit as st

def main():
    st.markdown("# 🧬 Cancer Genomics Classification")
    st.markdown("**Researchers:** Christina Xu (cjxu@bu.edu), Khanh Nguyen (khanhng@bu.edu)")
    st.markdown("## About")
    st.markdown("""
                The goal of this web application is to classify different cancer types and distinguish between benign and malignant 
                cancer tissue samples with a focus on developing novel feature transformations. It is a automated pipeline for 
                the fundamental steps of a AI/ML pipeline from data preprocessing, model training, to model evaluation on test data.
                """)
    st.markdown("## How to Use")
    st.markdown("""
                1. Click on the `Preprocess Data` tab in the sidebar. Select the cancer type you are interested in analyzing and then click on the `Prepare Data` button to transform
                   the data. 
                2. Next, click on the `Train Model` tab in the sidebar. Select the number of epochs you want to train the model and then click on the `Train Model` button.
                3. Lastly, click on the `Cancer Classification` tab to use your trained model to make an inference on test data. 
                """)
    st.markdown("## Source Code")
    st.markdown("""
                This is an open source project completed by the Boston University Machine Learning Research Group (BU MLRG) over summer 2023.
                All source code and documentation can be found on [GitHub](https://github.com/ml-sum23/genomic-cancer-classification).
                """)
    
if __name__ == '__main__':
    main()