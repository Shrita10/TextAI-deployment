from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
from doc3 import training_doc3

#change text to all lowercase and convert to a list of strings
cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
tokens = word_tokenize(cleaned)
#text sequences will be lists of length "train_len" where each element is a word
train_len = 4
text_sequences = []

#convert text into text sequences
for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequences.append(seq)

sequences = {}
count = 1
for i in range(len(tokens)):
    if tokens[i] not in sequences:
        sequences[tokens[i]] = count
        count += 1

#tokenizer assigns an integer value for each word, that value being determined by the frequency of that word
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences) 


vocabulary_size = len(tokenizer.word_counts)+1

n_sequences = np.empty([len(sequences),train_len], dtype='int32')

for i in range(len(sequences)):
    n_sequences[i] = sequences[i]
    
#convert numerical sequences to one-hot vectors (0s and 1s) for training
train_inputs = n_sequences[:,:-1]
train_targets = n_sequences[:,-1]
train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
seq_len = train_inputs.shape[1]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(50,activation='relu'))
model.add(Dense(vocabulary_size, activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#train model for 100 epochs
model.fit(train_inputs,train_targets,epochs=100,verbose=1)
model.save("mymodel.h5")
model.save("tokenizer.h5")

#Testing
from tensorflow.keras.preprocessing.sequence import pad_sequences
input_text = "he is very"
print(input_text)
encoded_text = tokenizer.texts_to_sequences([input_text])[0]
pad_encoded = pad_sequences([encoded_text], maxlen=3, truncating='pre')
list_of_words =[]
for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
    pred_word = tokenizer.index_word[i]
    list_of_words.append(pred_word)
    
first_word = list_of_words[0]
second_word = list_of_words[1]
third_word = list_of_words[2]

print(first_word, second_word, third_word)