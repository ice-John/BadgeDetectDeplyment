from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import numpy as np

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    tokenizer = pickle.load(open('tokenizer.p','rb'))
    model = load_model('model0.h5')
    
    

    if request.method == 'POST':
        message = request.form['message']
        X_tokens = tokenizer.texts_to_sequences([message])
        X_pad = pad_sequences(X_tokens, maxlen = 100, padding = 'post')
        #X_pad = X_pad.astype(np.float64)
        result_ls = list(model.predict_proba(X_pad))[0]
        #result_ls = result_ls.astype(np.float64)
        result = {'one':0,'two':0,'three':0,'four':0,'five':0,'six':0,'seven':0,'eight':0,'nine':0,'ten':0}
        i=0
        for k in result.keys():
            result[k] = result_ls[i]
        #     print(result[k], i)
            i=i+1

    return render_template('result.html',prediction = result)



if __name__ == '__main__':
    app.run(debug=False,threaded=False,host='0.0.0.0')
