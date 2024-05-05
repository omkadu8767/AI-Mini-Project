import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Load the database 
data = pd.read_csv('mail_data.csv')

# Replace 'spam' and 'ham' with 1 and 0 respectively
data['Category'] = data['Category'].map({'spam': 1, 'ham': 0})

# Split the data into features (email text) and labels (spam or not spam)
X = data['Message']
y = data['Category']

# Create a CountVectorizer to convert email text into a matrix of token counts
count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(X)

# Transform the count matrix into TF-IDF (Term Frequency-Inverse Document Frequency) representation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Train a Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

def predict_spam(text):
    # Convert input text into TF-IDF representation
    text_counts = count_vectorizer.transform([text])
    text_tfidf = tfidf_transformer.transform(text_counts)
    
    # Make prediction
    prediction = classifier.predict(text_tfidf)
    
    # Interpret prediction
    if prediction[0] == 1:
        return "SPAM"
    else:
        return "NOT SPAM"

# User input
user_text = input("Enter the email text: ")
result = predict_spam(user_text)
print("Prediction:", result)
