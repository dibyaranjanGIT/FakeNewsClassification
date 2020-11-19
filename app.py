# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 22:48:22 2020

@author: Dibyaranjan
"""


from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pickle

# load the model from disk
filename = 'GridClassifier.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('CV_tranformer.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]

		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)