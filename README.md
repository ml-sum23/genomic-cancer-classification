# Cancer Genomics Classification

The objective of this project is to discover novel feature mappings on cancer gene expression data for binary classification tasks. This repository documents the work of Boston University's Machine Learning Research Group over summer 2023, advised by Dr. Mark Kon. Our reasearch paper, in collaboration with the stock price prediction team, can be found [here.](https://docs.google.com/document/d/1RruGtprP9f7ZG3CShdeFrcWO6nJ7v5mWH2FGThk7aLM/edit?usp=sharing)

## Repository Structure
```
├── data 
├── models
├── src <- Source code for web application
├── streamlit <- Web application pages  
└── README.md <- You are here
```

## How to Deploy Web App
1. We use [Streamlit](https://streamlit.io/), an open-source app framework, to develop our web application. Please make sure you have Streamlit installed before you proceed. To install, run:

    ```
    pip install streamlit
    ```

    If the following command returns sensible results then Streamlit is installed:

    ```
    streamlit hello
    ```

2. Navigate to the `streamlit` subdirectory
    ```
    cd streamlit
    ```
3. To deploy the web application in your local browser, run:
    ```
    streamlit run Home.py
    ```

4. To stop the web application, on your keyboard, press:
    ```
    ctrl+c
    ```

## Group Members
- Khahn Nguyen (khanhng@bu.edu)
- Christina Xu (cjxu@bu.edu)

## Contact Info

Please contact Christina Xu (cjxu@bu.edu) for any concerns/feedback/questions.
