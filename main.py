import keras
import numpy as np
import pandas as pd
import streamlit as st
import askr_funcs as af

#Import data and create title and subheader
X = pd.read_csv('data/columns.csv')
v1 = pd.read_csv('data/vt/Vt_01.csv')
v2 = pd.read_csv('data/vt/Vt_02.csv')
v3 = pd.read_csv('data/vt/Vt_03.csv')
v4 = pd.read_csv('data/vt/Vt_04.csv')
Vt = pd.concat([v1,v2,v3,v4], axis=0)
Vt = Vt.values
columns = X.columns
model = keras.models.load_model('data/final_model')

st.title('Ask Reddit Post Score Classification Prediction ')
st.subheader('Welcome! Here you can input a potential title for your Ask Reddit post, and the model created will be able to tell you'
             ' whether or not it will get a low, medium, or high amount of upvotes!')

#Search and Select input
user_in = st.text_input(label='Type your potential post here!', help='Make sure to include a question mark \'?\'')

#When button is clicked it turns true and everything below happens
if st.button('Predict'):
    vec_user_in = af.user_in(user_in, columns)
    prediction = af.get_prediction(vec_user_in, model, Vt)
    st.text("")
    if prediction == 0:
        st.subheader('It is unlikely your post will get more than 100 upvotes.')
    elif prediction == 1:
        st.subheader('We predict your post will get somewhere between 100-1000 upvotes')
    elif prediction == 2:
        st.subheader('Our model says that your post will get more than 1000 upvotes!!')
    else:
        st.subheader('Sorry, there\'s something wrong going on behind the scenes. Contact Arran so he can take a look!')
    st.text("")

#spacing
for i in range(15):
    st.text("")

#Information
with st.expander("About the project and creator"):
    st.write("This project was created from data scraped through the Python Reddit API Wrapper (PRAW). The model used was a Multi Layer Perceptron Neural Net."
             " If you'd like to see the code that created the model and the UI, please click [here](https://github.com/arranwl/imdb_project)")
    st.write("About the creator:"
             " Arran Wass-Little is a third year student at the University of Florida undertaking a double"
             " major in Economics and Data Science. He's passionate about understanding the world through data"
             " whether through building tools or completing valuable research. If you'd like to contact him,"
             " please reach out to him at arranwasslittle@ufl.edu")