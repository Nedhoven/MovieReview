#!/usr/bin/env python
# coding: utf-8

# # Method 3: CNN

# In this part of the report, we examin Convolutional Neural Network (CNN) to try to predict the outcome of a movie review as a "Positive" or a "Negative" review. Please make sure you have the data set (consisting of positive reviews and negative reviews in the described foldres).

# ### Preprocessing

# As the first step, we assing the paths of the different sets of data.

# In[1]:


import numpy as np


train_path_pos = './Data/Train/pos/'
train_path_neg = './Data/Train/neg/'
test_path_pos = './Data/Test/pos/'
test_path_neg = './Data/Test/neg/'

np.random.seed(0)


# Now, we try to construct data structures appropriate for learning methods (a data frame) with the text (i.e., reviews) and the label (i.e., positive labeled as `1` and negative labeled as `0`)

# In[2]:


import os
import random as rd
from pandas import DataFrame


def read_all_files(path: str, label: int) -> dict:
    """reading all the txt files into a dictionary including the labels"""
    folder = os.listdir(path)
    res = {}
    for file in folder:
        text = open(path + file).readline().lower()
        res[text] = label
    return res

def concat_all(dict_1: dict, dict_2: dict) -> DataFrame:
    """creating a data frame"""
    temp = dict()
    temp.update(dict_1)
    temp.update(dict_2)
    keys = list(temp.keys())
    rd.shuffle(keys)
    target = []
    for key in keys:
        target.append(temp[key])
    ans = DataFrame({'text': keys, 'label': target})
    return ans


# Using the functions defined above, we can now create the training set.

# In[3]:


dict_pos = read_all_files(path=train_path_pos, label=1)
dict_neg = read_all_files(path=train_path_neg, label=0)
train_data = concat_all(dict_pos, dict_neg)
train_data.head()


# And the testing data set.

# In[4]:


dict_pos = read_all_files(path=test_path_pos, label=1)
dict_neg = read_all_files(path=test_path_neg, label=0)
test_data = concat_all(dict_pos, dict_neg)
test_data.head()


# After creating the data structure for training and testing sets, we need a way to process the natural language. One can do that using Natural Language Tool-Kit (`nltk`).

# In[5]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

stop_words = set(stopwords.words('english'))
word_table = str.maketrans('', '', punctuation)

def clean_text(text: str) -> list:
    """cleaning given text using natural language processing"""
    word_list = word_tokenize(text)
    word_list = [w for w in word_list if w.isalpha()]
    word_list = [w.translate(word_table) for w in word_list]
    word_list = [w for w in word_list if not w in stop_words]
    word_list = [w for w in word_list if len(w) > 1]
    return word_list


# Using the methods we defined above, we can process the train set.

# In[6]:


text_list = train_data['text']
rev_train = []
for text in text_list:
    words = clean_text(text=text)
    rev_train.append(words)
len(rev_train)


# We process the test set in the same manner we processed the train set.

# In[7]:


text_list = test_data['text']
rev_test = []
for text in text_list:
    words = clean_text(text=text)
    rev_test.append(words)
len(rev_test)


# Let us create a data frame to hold the length of each review.

# In[8]:


def rev_length(rev_list: list) -> DataFrame:
    """creating a data frame from the length of each review"""
    length = DataFrame([len(rev) for rev in rev_list])
    length.columns = ['length']
    return length


# Now, we can see the length of each review in the train set.

# In[9]:


len_train = rev_length(rev_train)
len_train.head()


# Let us take a look at the describtion of the train set lengths.

# In[10]:


len_train.describe()


# And also the length of each review in the test set.

# In[11]:


len_test = rev_length(rev_test)
len_test.head()


# And the test set describtion.

# In[58]:


len_test.describe()


# As it can be seen, most of the data lay before the length 141 (75%). We have to define `max_length` to consider the review for the training process.

# In[59]:


max_length = 256
max_length


# We can define a minimum length for the test set as well.

# In[60]:


max_length_test = 256
max_length_test


# Let us create a dictionary of the words and their frequency for further assessment.

# In[61]:


def create_freq(words_list: list) -> dict:
    """creating a dictionary of words frequency"""
    freq = {}
    for w in words_list:
        if w in freq:
            freq[w] += 1
        else:
            freq[w] = 1
    return freq


# Using the function above, we can count the words in the train set.

# In[62]:


import itertools as itr


words_train = itr.chain.from_iterable(rev_train)
freq_train = create_freq(words_list=words_train)
len(freq_train)


# In the same manner, we can count the words in the test set.

# In[63]:


words_test = itr.chain.from_iterable(rev_test)
freq_test = create_freq(words_list=words_test)
len(freq_test)


# The most frequent words are important. Therefore, we need to sort the dictionaries.

# In[64]:


import operator as op


freq_train_list = list(reversed(sorted(freq_train.items(), key=op.itemgetter(1))))
freq_test_list = list(reversed(sorted(freq_test.items(), key=op.itemgetter(1))))


# After having the reviews in the form of "bag of words", we need to assing them to numbers as well (e.g., most common word, second most common word, etc.) and we only keep count up to `cap` different words.

# In[65]:


cap = 7000
word_id_train = dict()
id_word_train = dict()
word_id_test = dict()
id_word_test = dict()
for i in range(0, cap):
    word_id_train[freq_train_list[i][0]] = i
    id_word_train[i] = freq_train_list[i][0]
    word_id_test[freq_test_list[i][0]] = i
    id_word_test[i] = freq_test_list[i][0]

len(word_id_train)


# ### Training and Evaluating

# In order to train the neural network, we need to create features based on the frequenct words that appear in the context of each review.

# In[68]:


def get_freq_data(rev: list) -> list:
    """getting the frequency list of the reviews"""
    ans = []
    for word in rev:
        if word in word_id_train:
            ans.append(word_id_train[word])
    return ans


# After defining the function above, we can create `x_set` and `y_set` for training purposes.

# In[69]:


x_set = []
y_set = []

for i in range(0, len(rev_train)):
    rev = get_freq_data(rev=rev_train[i])
    if len(rev) <= max_length_train:
        x_set.append(rev)
        y_set.append(train_data['label'][i])
x_set = np.asarray(x_set)
y_set = np.asarray(y_set)


# Since not all the reviews have the same size, we need to pad them so the appear in the same size.

# In[71]:


import keras
from keras.preprocessing import sequence

x_set = sequence.pad_sequences(x_set, maxlen=max_length, value=0)
(x_set.shape, y_set.shape)


# Now, we need to repoeat the process for the testing set as weel.

# In[73]:


x_test = []
y_test = []

# rev_small_test = rd.sample(rev_test, 5000)

for i in range(0, len(rev_test)):
    rev = get_freq_data(rev=rev_test[i])
    if len(rev) <= max_length_test:
        x_test.append(rev)
        y_test.append(test_data['label'][i])
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
x_test = sequence.pad_sequences(x_test, maxlen=max_length, value=0)
(x_test.shape, y_test.shape)


# After preparing the data for our mode, we can define the model using `keras`.

# In[74]:


from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout,Conv1D,Flatten
from keras.layers import Embedding

model = Sequential()
model.add(Embedding(cap, 32, input_length=max_length_train))
model.add(Conv1D(128, 3, padding='same'))
model.add(Conv1D(64, 3, padding='same'))
model.add(Conv1D(32, 2, padding='same'))
model.add(Conv1D(16, 2, padding='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# For the last layers, we chose `sigmoid` as the activation function, since we are classifying the reviews as positive or negative.

# In[75]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# The loss function is defined as `binary_crossentropy` and our metric to examin is `accuracy`. It is time to run the model on the data.

# In[76]:


model.fit(x_set, y_set, epochs=5, batch_size=64)


# In[77]:


score = model.evaluate(x_test, y_test, batch_size=64)
score[1]


# As it can be seen, after `epochs = 5`, the accuracy on the training data set is `% 98.48`, and the accuracy on the testing data set is `% 84.24`.
