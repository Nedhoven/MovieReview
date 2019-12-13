#!/usr/bin/env python
# coding: utf-8

# # Method 1: DNN

# First, we investigate Deep Neural Networks (DNN) on the IMDB movie reviews data set to see if we can predict the "Positive" or "Negative" labels for the reviews.

# ### Preprocessing

# In[23]:


import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import mltools as ml
import math as math
from numpy import asmatrix as arr
from imp import reload
import tarfile


# In[24]:


import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
dataset = tf.keras.utils.get_file(fname="aclImdb.tar.gz",
                                  origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
                                  extract=True)


# Downloading data from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# he dataset consists of IMDB movie reviews labeled by positivity from 1 to 10. The task is to label the reviews as negative or positive.

# In[25]:


directory = dataset


# In[26]:


# Load all files from a directory in a DataFrame.

def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# In[27]:


# Merge positive and negative examples, add a polarity column and shuffle.

def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# In[28]:


train_df = load_dataset(os.path.join(os.path.dirname(dataset),"aclImdb", "train"))
test_df = load_dataset(os.path.join(os.path.dirname(dataset), "aclImdb", "test"))


# In[29]:


# Training input on the whole training set with no limit on training epochs.
train_input_fn =tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
# Prediction on the test set.
predict_test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)


# In[30]:


embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")


# ### Training and Evaluating

# In[31]:


# For classification we can use a DNN Classifier 
estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.optimizers.Adam(learning_rate=0.003))


# Train the estimator for a reasonable amount of steps.

# In[32]:


estimator.train(input_fn=train_input_fn, steps=1000);


# Run predictions for both training and test set.

# In[33]:


train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))


# We can visually check the confusion matrix to understand the distribution of misclassifications.

# In[34]:


def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

LABELS = [
    "negative", "positive"
]

# Create a confusion matrix on training data.
with tf.Graph().as_default():
    cm = tf.compat.v2.math.confusion_matrix(train_df["polarity"], 
         get_predictions(estimator, predict_train_input_fn))
    with tf.compat.v1.Session() as session:
         cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS);
plt.xlabel("Predicted");
plt.ylabel("True")


#  we can improve the accuracy by tuning the meta-parameters like the learning rate or the number of steps, especially if we use a different module. A validation set is very important if we want to get any reasonable results, because it is very easy to set-up a model that learns to predict the training data without generalizing well to the test set.
