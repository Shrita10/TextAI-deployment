# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 03:49:28 2021

@author: Shrita
"""

import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from doc3 import training_doc3
from flask_restful import Resource, Api, reqparse
from tensorflow.keras.models import load_model

WORDS_USED = {}
parser = reqparse.RequestParser()

app = Flask(__name__)
api = Api(app)
model = load_model('mymodel.h5', compile = False)

tokenizer = load_model('tokenizer.h5')

class WordList(Resource):
    def get(self):
          return WORDS_USED
    def post(self):
          parser.add_argument("word_used")
          parser.add_argument("previous_words")
          parser.add_argument("other_options")
          args = parser.parse_args()
          word_id = int(max(WORDS_USED.keys())) + 1
          word_id = '%i' % word_id
          WORDS_USED[word_id] = {
            "word_used": args["word_used"],
            "previous_words": args["previous_words"],
            "other_optins": args["other_options"],
          }
          return WORDS_USED[word_id], 201


class Word(Resource):
    def get(self, word_id):
          if word_id not in WORDS_USED:
              return "Not found", 404
          else:
              return WORDS_USED[word_id]
          
    def put(self, word_id):
          parser.add_argument("word_used")
          parser.add_argument("previous_words")
          parser.add_argument("other_options")
          args = parser.parse_args()
          if word_id not in WORDS_USED:
            return "Record not found", 404
          else:
            word = WORDS_USED[word_id]
            word["word_used"] = args["word_used"] if args["word_used"] is not None else word["word_used"]
            word["previous_words"] = args["previous_words"] if args["previous_words"] is not None else word["previous words"]
            word["other_options"] = args["other_options"] if args["other_options"] is not None else word["other_options"]
            return word, 200
        
    def delete(self, word_id):
          if word_id not in WORDS_USED:
              return "Not found", 404
          else:
              del WORDS_USED[word_id]
              return '', 204

api.add_resource(WordList, '/words/')
api.add_resource(Word, '/words/<word_id>')

counter = 1
@app.route('/')

def home():
    return render_template('html1.html')

@app.route('/predict2',methods = ['POST'])

def predict2():
        global counter
        global WORDS_USED
        input_text = request.form['ttext']
        cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
        tokens = word_tokenize(cleaned)
        train_len = 1
        text_sequences = []  
        for i in range(train_len,len(tokens)):
            text_sequences.append(tokens[i])
        sequences = {}
        count = 1
        for i in range(len(tokens)):
            if tokens[i] not in sequences:
                sequences[tokens[i]] = count
                count += 1
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_sequences)
        input_text = input_text.strip().lower()
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
        list_of_words =[]
        for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
            pred_word = tokenizer.index_word[i]
            list_of_words.append(pred_word)
        first_word = list_of_words[0]
        second_word = list_of_words[1]
        third_word = list_of_words[2]
        WORDS_USED[str(counter)] = {'word_used': None, 'previous_words': request.form['ttext'], 'other_options': [first_word, second_word, third_word]}
        counter += 1
        return render_template('html1.html',prediction_text1 = first_word , prediction_text2 =  second_word, prediction_text3 = third_word)

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    