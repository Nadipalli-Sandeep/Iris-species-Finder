pip install scikit-learn
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

st.write('''
#Simple iris Flower prediction App

This app predicts the iris flower type!
''')
st.sidebar.header('User input parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal_length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal_width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal_length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal_width', 0.1, 2.5, 0.2)
    data = {'Sepal_length':sepal_length,
            'Sepal_width':sepal_width,
            'Petal_length':petal_length,
            'Petal_width':petal_width
    }
    features = pd.DataFrame(data, index = [0])
    return features
df = user_input_features()
st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])


st.subheader('Prediction Probability')
st.write(prediction_proba)
