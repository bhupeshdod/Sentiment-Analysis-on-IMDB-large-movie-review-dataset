# importing required packages
import numpy as np
import pandas as pd
import os
from keras.layers import Dense, Dropout
from nltk.corpus import stopwords
import re
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

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
neg_train = np.array(load_1('./data/aclImdb/train/neg'))
pos_train = np.array(load_1('./data/aclImdb/train/pos'))
neg_train_label = np.array([0]*12500)
pos_train_label = np.array([1]*12500)
train_label = np.concatenate((neg_train_label, pos_train_label)).reshape(25000,1)
train = np.concatenate((neg_train, pos_train)).reshape(25000,1)
train_dataset = pd.DataFrame(np.concatenate((train, train_label), axis=1))

#Function to remove all special characters from trainset.
def preprocess_reviews(reviews):
    tokens = re.compile("[.;:!#\'?,\"()\[\]]||(\-)|(\/)(<br\s*/><br\s*/>)")
    return [tokens.sub("", line.lower()) for line in reviews]
X_train = preprocess_reviews(train.reshape(25000,))

#Function to remove all stopwords from trainset.
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(' '.join([word for word in review.split() if word not in english_stop_words]))
    return removed_stop_words
X_train = remove_stop_words(X_train)

#Function to tokenize all the text data (Trainset data) into numeric data. In this function we are storing keras tokenizer in .pkl to use it later in testset tokenization.
tokenize = tf.keras.preprocessing.text.Tokenizer()
tokenize.fit_on_texts(X_train)
pickle.dump(tokenize, open('./models/tokenizer.pkl','wb'))    
X_train = tokenize.texts_to_sequences(X_train)
length = [len(s) for s in (X_train)]
max_length = max(length)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_train, X_val, y_train, y_val = train_test_split(X_train, train_label, test_size=0.2, random_state=42)

#Model for processing and classification of data. In this code, we are saving model in the format .h5 so that we can use it later to evaluate on testset.
model = Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(tokenize.word_index)+1, output_dim=100, input_length=max_length))
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="leaky_relu"))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(Dropout(0.1))
model.add(Dense(512, activation='leaky_relu'))
model.add(Dropout(0.1))
model.add(Dense(256, activation='leaky_relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='leaky_relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.experimental.Adam(learning_rate=0.001), metrics='accuracy')
history_model = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_data=(X_val, y_val), verbose=2)
model.summary()
model.save('./models/20990294_NLP_model.h5')

#Code for plotting Training vs validation loss for each epoch.
plt.figure(figsize = (15,5.5))
plt.subplot(121)
plt.plot(history_model.history['val_loss'],label="val loss", c = 'r', marker = '.')
plt.plot(history_model.history['loss'],label="train loss", c = 'b', marker = '.')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.grid()
plt.legend()

#Code for plotting Training vs validation accuracy for each epoch.
plt.subplot(122)
plt.plot(history_model.history['val_accuracy'], label="val accuracy", c = 'r', marker = '.')
plt.plot(history_model.history['accuracy'], label="Train accuracy", c = 'b', marker = '.')
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation Accuracy')
plt.grid()
plt.legend()
plt.show()
