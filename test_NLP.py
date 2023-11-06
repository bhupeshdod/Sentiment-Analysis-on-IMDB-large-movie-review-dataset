# import required packages
import numpy as np
import pandas as pd
import os
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
import re
import pickle
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

#Loading file to read and return review inside the file
def load(file):
    file_1 = open(file, 'r', encoding="utf8")
    text = file_1.read()
    file_1.close()
    return text

#Loading files of each review and appending them together to create a list
def load_1(folder):
    dic=[]
    for i in os.listdir(folder):
        if not i.endswith(".txt"):
            continue
        path = folder+'/'+i
        dic.append(load(path))
    return dic

#Preprocessing steps to create train dataset with it's labels
neg_test = np.array(load_1('./data/aclImdb/test/neg'))
pos_test = np.array(load_1('./data/aclImdb/test/pos'))
neg_test_label = np.array([0]*12500)
pos_test_label = np.array([1]*12500)
y_test = np.concatenate((neg_test_label, pos_test_label)).reshape(25000,1)
test = np.concatenate((neg_test, pos_test)).reshape(25000,1)
test_dataset = pd.DataFrame(np.concatenate((test, y_test), axis=1))

#Function to remove all special characters from trainset.
def preprocess_reviews(reviews):
    tokens = re.compile("[.;:!#\'?,\"()\[\]]||(\-)|(\/)(<br\s*/><br\s*/>)")
    return [tokens.sub("", line.lower()) for line in reviews]
X_test = preprocess_reviews(test.reshape(25000,))

#Function to remove all stopwords from trainset.
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(' '.join([word for word in review.split() if word not in english_stop_words]))
    return removed_stop_words
X_test = remove_stop_words(X_test)

#We are using same tokenizer to tokenize the testset data that we created and stored as .pkl file from train_NLP.py file.
tokenize = pickle.load(open("./models/tokenizer.pkl",'rb'))
X_test = tokenize.texts_to_sequences(X_test)
max_length = 1459
X_test = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

#We are using same model that we trained and stored in .h5 format from train_NLP.py file.
model = tf.keras.models.load_model('./models/20990294_NLP_model.h5')
acc = model.evaluate(X_test, y_test)
test_acc = acc[1]
print('Accuracy on test set:', test_acc*100)