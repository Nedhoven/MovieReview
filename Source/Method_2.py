#!/usr/bin/env python
# coding: utf-8

# # Method 2: LSTM and Logisitc Regression

# In this run through, we first build a corpus preprocessing to remove punctuation and stop words. Then, we implement an LSTM using keras to train a model. Obviously, LSTM is an overkill for a tiny dataset like this, however, this will lay a baseline to show how a much simpler model can achieve a similar accuracy (i.e. Logistic Regression). In addition, we analyze both TF-IDF and CountVectorization with bigrams and unigrams as well.

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from sklearn.decomposition import PCA
import glob
import tensorflow.keras.preprocessing.text as Preprocess
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

np.random.seed(0)


# ### Preprocessing

# First, we try to load the data.

# In[2]:


def load_dataset(path, label):
    x = []
    y = []
    
    filenames = glob.glob(path + "*.txt")
    
    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as file:
            x.append(file.read())
        y.append(label)
    
    return x, y


# In[3]:


x_train, y_train, x_test, y_test = [], [], [], []
x, y = load_dataset("aclImdb/train/pos/", 1)
x_train.extend(x)
y_train.extend(y)

x, y = load_dataset("aclImdb/train/neg/", 0)
x_train.extend(x)
y_train.extend(y)

x, y = load_dataset("aclImdb/test/pos/", 1)
x_test.extend(x)
y_test.extend(y)

x, y = load_dataset("aclImdb/test/neg/", 0)
x_test.extend(x)
y_test.extend(y)


# After loading the dataset, we start by removing some stop words from the dataset. We do this by using NLTK's base stop word dataset. Just as a preprocessing step, we divide the reviews into word sequences as well. This is both for convenience to remove stop words and later on build our corpus.

# In[4]:


stop_words = set(stopwords.words('english'))


# In[22]:


x_train_ws = [Preprocess.text_to_word_sequence(review) for review in x_train]
x_test_ws = [Preprocess.text_to_word_sequence(review) for review in x_test]


# In[23]:


def remove_stopwords(data):
    for i in range(len(data)):
        data[i] = [word for word in data[i] if word not in stop_words]
    return data


# In[25]:


x_train_ws = remove_stopwords(x_train_ws)
x_test_ws = remove_stopwords(x_test_ws)


# We next build the corpus and get the tokenized words converted to represent a unique integer index.

# In[8]:


x_train_re = [' '.join(i) for i in x_train_ws]
x_test_re = [' '.join(i) for i in x_test_ws]


# In[9]:


vectorizer = CountVectorizer(analyzer='word', max_features=8000)
x_train_v = vectorizer.fit_transform(x_train_re)
unique_words = vectorizer.get_feature_names()


# In[10]:


x_train_i = []
x_test_i = []

for review in x_train_ws:
    x_train_i.append([vectorizer.vocabulary_[word]+1 for word in review if word in vectorizer.vocabulary_])

for review in x_test_ws:
    x_test_i.append([vectorizer.vocabulary_[word]+1 for word in review if word in vectorizer.vocabulary_])


# ### Training and Evaluating

# ##### LSTM
# Before feeding the data into neural network, we have to truncate or pad the reviews to make sure they are of certain length. This is done by analyzing the dataset to visualize the review lengths, and thereby picking fixed length. For our classifier, we decided to pick a length of 300.

# In[27]:


data_lengths = [len(x) for x in x_train_i]

figure = plt.figure(figsize=(8, 8))
plt.hist(data_lengths, bins=100)
plt.title("Review length")
plt.show()


# In[34]:


x_train_truncated = keras.preprocessing.sequence.pad_sequences(x_train_i, maxlen=300)
x_test_truncated = keras.preprocessing.sequence.pad_sequences(x_test_i, maxlen=300)


# In[35]:


x_train_n = np.array(x_train_truncated)
y_train_n = np.array(y_train)
x_test_n = np.array(x_test_truncated)
y_test_n = np.array(y_test)

print(x_train_n.shape)
print(y_train_n.shape)


# In[45]:


model = Sequential()
model.add(Embedding(len(vectorizer.vocabulary_)+1, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# In[46]:


ckpt_callback = ModelCheckpoint('imdb.h5', save_best_only=True, monitor='val_loss', mode='min')

model.fit(x_train_n, y_train_n, batch_size=64, shuffle=True,
          validation_split=0.15, epochs=3, callbacks=[ckpt_callback])


# In[48]:


model = load_model('imdb.h5')


# In[49]:


(loss, accuracy) = model.evaluate(x_test_n, y_test_n, verbose=0, batch_size=128)

print("Test loss : ", loss)
print("Test accuracy : ", accuracy)

(loss, accuracy) = model.evaluate(x_train_n, y_test_n, verbose=0, batch_size=128)

print("Train loss : ", loss)
print("Train accuracy : ", accuracy)


# The model achieved a test accuracy of 85.98% and training accuracy of 92.71%. Leaving it running for longer yielded a much lower training loss than validation loss. Therefore, we decided to halt the model only after two epochs.

# ##### Logistic Regression

# We first use CountVectorization and then TF-IDF with unigrams.

# In[51]:


x_train_v = vectorizer.transform(x_train_re)
x_test_v = vectorizer.transform(x_test_re)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train_v, y_train)

accuracy = logistic_regression.score(x_train_v, y_train)
print("Logistic regression (Count vectorizer, Unigram) train accuracy : ", accuracy)

accuracy = logistic_regression.score(x_test_v, y_test)
print("Logistic regression (Count vectorizer, Unigram) test accuracy : ", accuracy)


# In[52]:


idf_vectorizer = TfidfVectorizer(analyzer='word')
x_train_idf = idf_vectorizer.fit_transform(x_train_re)
x_test_idf = idf_vectorizer.transform(x_test_re)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train_idf, y_train)

accuracy = logistic_regression.score(x_train_idf, y_train)
print("Logistic regression (TfIdf, Unigram) train accuracy : ", accuracy)

accuracy = logistic_regression.score(x_test_idf, y_test)
print("Logistic regression (TfIdf, Unigram) test accuracy : ", accuracy)


# Now, we include Bigrams.

# In[54]:


vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))

x_train_v = vectorizer2.fit_transform(x_train_re)
x_test_v = vectorizer2.transform(x_test_re)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train_v, y_train)

accuracy = logistic_regression.score(x_train_v, y_train)
print("Logistic regression (Count vectorizer, Bigram) train accuracy : ", accuracy)

accuracy = logistic_regression.score(x_test_v, y_test)
print("Logistic regression (Count vectorizer, Bigram) test accuracy : ", accuracy)


# In[56]:


idf_vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
x_train_idf = idf_vectorizer2.fit_transform(x_train_re)
x_test_idf = idf_vectorizer2.transform(x_test_re)

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train_idf, y_train)

accuracy = logistic_regression.score(x_train_idf, y_train)
print("Logistic regression (TfIdf, Bigram) train accuracy : ", accuracy)

accuracy = logistic_regression.score(x_test_idf, y_test)
print("Logistic regression (TfIdf, Bigram) accuracy : ", accuracy)


# Adding bigrams improved the accuracy for the count vectorization but not for TF-IDF.

# And here is the final comparison and the results.

# In[59]:


train_acc = [0.92716, 0.98576, 0.93656, 1.0, 0.96264]
test_acc = [0.8598, 0.85044, 0.88264, 0.88644, 0.88012]
labels = ['LSTM', 'CV-UNIGRAM+LR', 'TF-IDF-UNIGRAM+LR', 'CV-BIGRAM+LR', 'TF-IDF-BIGRAM+LR']

x = np.arange(len(labels))
width = 0.35

figure, ax = plt.subplots(figsize=(12,12))
rects1 = ax.bar(x - width/2, train_acc, width, label='Train')
rects2 = ax.bar(x + width/2, test_acc, width, label='Test')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracies for 5 models')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# As can be seen from the figure, a simpler model also achieves a similar accuracy with a little bit more complex model. In fact, TF-IDF with both Unigram and Bigram seems to be doing better.
