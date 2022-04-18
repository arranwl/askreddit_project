import keras
import numpy as np
import pandas as pd
import streamlit as st
import askr_funcs as af

#Import data and create title and subheader
X = pd.read_csv('X.csv')
Vt = pd.read_csv('Vt.csv')
Vt = Vt.values
columns = X.columns
model = keras.models.load_model('final_model')

vec_user_in = af.user_in("What is your name?", columns)
prediction = af.get_prediction(vec_user_in, model, Vt)

if prediction == 0:
    print('It is unlikely your post will get more than 100 upvotes.')
elif prediction == 1:
    print('We predict your post will get somewhere between 100-1000 upvotes')
elif prediction == 2:
    print('Our model says that your post will get more than 1000 upvotes!!')
else:
    print('Sorry, there\'s something wrong going on behind the scenes. Contact Arran so he can take a look!')