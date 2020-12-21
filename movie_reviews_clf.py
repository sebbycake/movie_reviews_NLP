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
# as the tab was not consistent in the header initially
df = pd.read_csv("train.tsv", sep='\t', header=0)

# Print the head of df
print("Raw: \n", df.head(10), '\n')


# Exploratory data analysis (EDA) - for high level summary of data
size_of_data = len(df)
positive_reviews_count = len(df[df['sentiment'] == 1])
negative_reviews_count = len(df[df['sentiment'] == 0])
df_review_len = df['review'].map(lambda text: len(text))
shortest_review = min(df_review_len)
longest_review = max(df_review_len)
avg_review_length = round(df_review_len.mean())
std_review = round(df_review_len.std())

print('---Key Statistics---')
print(f'Size of data {size_of_data}\n') # 997
print(f'Number of positive reviews: {positive_reviews_count}\n') # 480
print(f'Number of negative reviews: {negative_reviews_count}\n') # 517
print(f'Shortest review: {shortest_review} chars\n') # 127 chars
print(f'Longest review: {longest_review} chars\n') # 5809 chars
print(f'Average length of review: {avg_review_length} chars\n') # 1317 chars
print(f'Standard deviation of review: {std_review} chars\n') # 959 chars

# This is a supervised learning classification task, 
# with sentiment as the response variable and review as the only feature.
# There are a total of 997 reviews in our dataset. 
# Among them, 480 are positive and 517 are negative, which is rather balanced. 
# The average length of reviews across the entire dataset is 1317 characters,
# with the smallest being 517 and the largest being 5809, which is a significant gap.
# Moreover, the standard deviation is 959 characters.
# A quick scanning of the reviews I found that there are many of them
# with HTML tags and special characters, which can be removed as they are meaningless in text analysis.
# Many names were also mentioned in the review.

# --------------------------------------

# Data cleaning and text preprocessing
# Short summary:
# Since reviews contain HTML tags, they are removed as they hold no meaning.
# Also, convert all reviews to lowercase as the model may treat capitalized form of two same words the same.
# Remove extra whitespaces as they are not needed.
# Perform lemmatization to ensure greater accuracy of meaning of words.

def clean_review(review):

    # 1. remove HTML tags
    soup = BeautifulSoup(review, "html.parser")
    no_html_tags = soup.get_text(separator=" ")

    # 2. convert to lowercase
    lowercase = no_html_tags.lower()

    # 3. remove extra whitespaces
    no_white_spaces = lowercase.strip()

    # 4. remove special characters
    no_spec_chars = re.sub('[^A-Za-z0-9]+', ' ', no_white_spaces)

    # 5. lemmatization: converting word to its base form
    lemmatizer = WordNetLemmatizer() 
    lemmatized_review = lemmatizer.lemmatize(no_spec_chars)

    return lemmatized_review

df['review'] = df['review'].map(lambda text: clean_review(text))

print("Cleaned: \n", df.head(10), '\n')


# --------------------------------------


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

