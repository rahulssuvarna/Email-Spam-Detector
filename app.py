
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
import joblib









# Load your dataset (update the path)
df = pd.read_csv('emails.csv')  # or your downloaded Kaggle dataset

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
display(df.head())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check class distribution
print("\nSpam vs Ham distribution:")
print(df['Prediction'].value_counts())

# Visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Prediction', data=df)
plt.title('Email Class Distribution')
plt.show()










# Initialize and train the model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()















# Separate features and target
X = df.drop(['Email No.', 'Prediction'], axis=1)  # Features (word counts)
y = df['Prediction']  # Target (1=spam, 0=ham)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)









# Combine all spam words
spam_text = ' '.join(list(spam_words.sort_values(ascending=False).head(50).index))

# Generate word cloud
plt.figure(figsize=(10,6))
spam_wc = WordCloud(width=800, height=400).generate(spam_text)
plt.imshow(spam_wc)
plt.axis('off')
plt.title('Spam Email Word Cloud')
plt.show()

# Combine all ham words
ham_text = ' '.join(list(ham_words.sort_values(ascending=False).head(50).index))

# Generate word cloud
plt.figure(figsize=(10,6))
ham_wc = WordCloud(width=800, height=400).generate(ham_text)
plt.imshow(ham_wc)
plt.axis('off')
plt.title('Ham Email Word Cloud')
plt.show()









# Save the trained model
joblib.dump(nb_model, 'spam_detector_model.pkl')

# Save the vectorizer (if you were using one)
# In this case we're using the existing word counts, but for new text we'd need this
# joblib.dump(vectorizer, 'count_vectorizer.pkl')

print("Model saved successfully!")








from sklearn.feature_extraction.text import CountVectorizer
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Download NLTK data (run once)
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing stopwords
    4. Stemming words
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize and remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    
    # Stem words
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Initialize vectorizer with same vocabulary as training data
vectorizer = CountVectorizer(vocabulary=X.columns)

def predict_spam(email_text):
    """
    Complete spam prediction pipeline for raw email text
    """
    # Preprocess the text
    processed_text = preprocess_text(email_text)
    
    # Vectorize using the same features as training
    email_counts = vectorizer.transform([processed_text])
    
    # Predict using our trained model
    prediction = nb_model.predict(email_counts)
    proba = nb_model.predict_proba(email_counts)
    
    # Format output
    result = "SPAM" if prediction[0] == 1 else "HAM"
    confidence = proba[0][1] if prediction[0] == 1 else proba[0][0]
    
    return f"{result} (confidence: {confidence:.2%})"

# Test with example emails
test_emails = [
    "Congratulations! You've won a $1000 prize! Click here to claim your money now!",
    "Hi John, just following up on our meeting tomorrow at 10am. Best regards, Sarah",
    "URGENT: Your account has been compromised. Verify your identity immediately!",
    "Here's the report you requested. Please review when you get a chance."
]

for email in test_emails:
    print("\nEmail:", email[:50] + "..." if len(email) > 50 else email)
    print("Prediction:", predict_spam(email))
