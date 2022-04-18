import numpy as np
import pandas as pd
import nltk
from tensorflow import keras

nltk.download('punkt')

def user_in(text, data_columns):
    snow = nltk.stem.SnowballStemmer('english')
    columns = data_columns.copy()
    output = [0]*(columns.size)
    user_tokens = nltk.tokenize.word_tokenize(text)
    cut = [snow.stem(word) for word in user_tokens]
    for word in cut:
        try:
            ind = columns.get_loc(word)
            output[ind] += 1
        except:
            continue
    output = np.array(output)
    output = output / np.sqrt(np.square(output).sum())
    output = output.reshape(1,output.size)
    return output

def get_prediction(vector, model, Vt):
    vector = vector[:,:1000]
    temp = ((vector@Vt)[0,:1000]).reshape(1,1000)
    probability = model.predict(temp)
    return np.argmax(probability, axis=1)[0]
