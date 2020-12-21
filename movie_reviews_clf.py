# Import the necessary modules
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# load data
# note: to re-tab the header before loading the file
# as the tab was not consistent in the header
df = pd.read_csv("train.tsv", sep='\t', header=0)

# Print the head of df
print(df.head(10))


# Data cleaning and text preprocessing

# 1. Remove HTML tags
def strip_html_tags(text):
    """
    remove HTML tags
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

df['review'] = df['review'].map(strip_html_tags)

# 2. Remove extra whitespaces
df['review'] = df['review'].map(lambda text: text.strip())

# 3. Remove special characters
df['review'] = df['review'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

# 4. Lemmatization: converting word to its base form
lemmatizer = WordNetLemmatizer() 
df['review'] = df['review'].map(lambda text: lemmatizer.lemmatize(text))


# Create a series to store the response variable (sentiment): y
y = df['sentiment']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data using only the 'review' column values: count_train 
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test data using only the 'review' column values: count_test 
count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# --------------------------------------
# Training and testing the "movie reviews sentiment" model with CountVectorizer

# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB();

# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
nb_pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: acc_score
nb_acc_score = metrics.accuracy_score(y_test, nb_pred)
print(f"MultinomialNB's Accuracy Score: {nb_acc_score}")

nb_f1_score = metrics.f1_score(y_test, nb_pred)
print(f"MultinomialNB's F1 Score: {nb_f1_score}")

print("")

# --------------------------------------
# Repeat with Random Forest Classifier model

# Instantiate a Random Forest classifier: nb_classifier
rf_classifier = RandomForestClassifier();

# Fit the classifier to the training data
rf_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
rf_pred = rf_classifier.predict(count_test)

# Calculate the accuracy score: acc_score
rf_acc_score = metrics.accuracy_score(y_test, rf_pred)
print(f"Random Forest's Accuracy Score: {rf_acc_score}")


rf_f1_score = metrics.f1_score(y_test, rf_pred)
print(f"Random Forest's F1 Score: {rf_f1_score}")


# --------------------------------------
# Hyperparameter Tuning

# Create a list of alphas: alphas
alphas = np.arange(0, 1, 0.1)

param_grid = {
    'alpha': alphas
}

clf = GridSearchCV(nb_classifier, param_grid, scoring='f1')

clf.fit(count_train, y_train)

print(f"Best Parameters: {clf.best_params_}")

alpha_value = clf.best_params_['alpha']


# --------------------------------------
# Re-train model with improved hyperparameter

new_nb_classifier = MultinomialNB(alpha=alpha_value);

# Fit the classifier to the training data
new_nb_classifier.fit(count_train, y_train)

# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: acc_score
nb_acc_score = metrics.accuracy_score(y_test, pred)
print(f"New MultinomialNB's Accuracy Score: {nb_acc_score}")

nb_f1_score = metrics.f1_score(y_test, pred)
print(f"New MultinomialNB's F1 Score: {nb_f1_score}")

