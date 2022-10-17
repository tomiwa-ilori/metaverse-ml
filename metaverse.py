"""

Goal: Detection of authentic news articles using ML and Auto-encoders  || Author: Israel Ilori || Date: October 2022

Our method of approach here is to first do some exploration on the dataset; to give us an idea of words which are contained in the corpus.
After which, we clean the text using a custom-built preprocessing function.
Next we use TF-IDF vectorizer to transform our text into vector matrices then 
Create a baseline algrithm using traditional ML. The ML of choice is a foundational LogisticRegression (LR) algorithm which is can be used for Binary Classification.

Summary:
    Using LR, the model seemed to perform well on the positive class but very poorly on the test set despite the dataset being fairly balanced.
    To improve the model:
        1. Extend the stopwords list, as the wordcloud showed commonly occuring words overlapping in both labels.
        3. Source for more training data for both classes, but with more focus on the negative class.
        2. To serve as a check, I would use an auto-encoder i.e HuggingFace's BERT to attempt to improve model performance. 

"""

# import necessary packages
import nltk
import string
import numpy as np
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from tqdm import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, classification_report




# read in data to pandas
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
label_data = pd.read_csv('labels.csv')

# drop duplicates values
train_data = train_data.drop_duplicates(subset=['text'])

# we are mostly concerned with the text coloumn as we plan to limit this solution to just it 
# hence, for this reason we drop the 1 empty row on the text column
train_data = train_data.dropna(subset='text')


""" Repeat the process for the test dataset """

# drop duplicates values
test_data = test_data.drop_duplicates(subset=['text'])

# drop empty rows for the text column
test_data = test_data.dropna(subset='text')


""" 

From observation, we notice that the test dataset is unlabeled. 
However, in our label.csv data, we see corresponding IDs to that of the test set, which have labellings
Therefore, we merge the the label.csv to test.csv on the id column 

"""

# left join labels.csv to test.csv on ID column
new_test_data = pd.merge(test_data, label_data, how="left", on=["id"])

# then check the shape
new_test_data.shape

"""  

Checking the data distribution of the training set using train_data['label'].value_counts(), 
we observe the dataset is fairly balanced. Next we proceed to cleaning the data.

"""

# cleaning function to remove 
#  stopwords,
#  punctuations, 
#  numrical values,
# ensure the text are english alphabets

remove_punctuation = str.maketrans('', '', string.punctuation)
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    uncased = text.translate(remove_punctuation).lower()
    tokens = [token for token in nltk.word_tokenize(uncased) 
                if len(token) > 1
                and not token in stop_words
                and not (token.isnumeric() and len(token) != 4)
                and (not token.isnumeric() or token.isalpha())]
    
    return " ".join(tokens)

# create a new column for the cleaned text in the training data
train_data['cleaned_text'] = train_data['text'].progress_apply(clean_text)

# repeat process for the test data
new_test_data['cleaned_text'] = new_test_data['text'].progress_apply(clean_text)


""" After the cleaning step, we perform a worccloud of the top 1000 frequently occuring words. 
    This aids in understanding what the vocubulary of the corpus is.
    To kick off, we build a wordcloud for the positive class label.
"""

# get the class label for the positive class
train_data_positive = train_data[train_data['label'] == 1]

# convert to series
train_data_positive_strings = pd.Series(train_data['cleaned_text']).str.cat(sep=' ')


# plot wordcloud
train_data_positive_wordcloud = WordCloud(  width=1600, stopwords=stop_words, height=800,
                        max_font_size=200, max_words=1000, collocations=False,
                        background_color='white').generate(train_data_positive_strings)

plt.figure(figsize=(20,20))
plt.imshow(train_data_positive_wordcloud, interpolation="bilinear")
plt.title('WordCloud showing the top 1000 words for the Positive Class Label')
plt.axis("off")
plt.show()


# get the class label for the negative class
train_data_negative = train_data[train_data['label'] == 0]

# convert to series
train_data_negative_strings = pd.Series(train_data['cleaned_text']).str.cat(sep=' ')


# plot wordcloud
train_data_negative_wordcloud = WordCloud(  width=1600, stopwords=stop_words, height=800,
                        max_font_size=200, max_words=1000, collocations=False,
                        background_color='white').generate(train_data_negative_strings)

plt.figure(figsize=(20,20))
plt.imshow(train_data_negative_wordcloud, interpolation="bilinear")
plt.title('WordCloud showing the top 1000 words for the Negative Class Label')
plt.axis("off")
plt.show()


""" Building the ML Algorithm """

# performing the multiclass classification

def run_prediction():
    X_train = train_data['cleaned_text']
    y_train = train_data['label']

    X_test = new_test_data['cleaned_text']
    y_test = new_test_data['label']

    # instantiate the vectorizer
    tfidf_vect = TfidfVectorizer(max_features=5000)

    X_train_vectorized = tfidf_vect.fit_transform(X_train)
    X_test_vectorized = tfidf_vect.fit_transform(X_test)

    # instantiate the LR Classifier with an ovr multi_class attribute: because the problem is a binary one
    model_linear = LogisticRegression(multi_class='ovr', penalty='l1', solver='liblinear').fit(X_train_vectorized, y_train)

    # make predictions on the validation set
    y_pred = model_linear.predict(X_test_vectorized)
  
    # classification report
    print (classification_report(y_test, y_pred))
