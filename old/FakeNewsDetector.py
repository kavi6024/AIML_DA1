import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize global variables
global_vectorizer = None
lr = None
dt = None
rf = None
gb = None

def load_data():
    # Read the datasets
    true_news = pd.read_csv('Large_True_News.csv')
    fake_news = pd.read_csv('Large_Fake_News.csv')
    
    # Remove duplicates
    true_news = true_news.drop_duplicates(subset=['text'])
    fake_news = fake_news.drop_duplicates(subset=['text'])
    
    # Add labels
    true_news['label'] = 1
    fake_news['label'] = 0
    
    # Combine datasets
    df = pd.concat([true_news, fake_news], ignore_index=True)
    
    # Remove any null values
    df = df.dropna(subset=['text', 'title'])
    
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Keep some punctuation as it might be indicative
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Tokenization and remove stopwords
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    # Add custom stopwords that might be too common in news
    custom_stops = {'news', 'say', 'said', 'tell', 'told'}
    stop_words.update(custom_stops)
    
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def prepare_data(df):
    # Combine title and text for better feature extraction
    df['content'] = df['title'] + ' ' + df['text']
    
    # Preprocess the content
    df['content'] = df['content'].apply(preprocess_text)
    
    # Split features and labels
    X = df['content']
    y = df['label']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def initialize_models():
    global lr, dt, rf, gb
    
    # Add regularization and class balancing
    lr = LogisticRegression(
        C=0.1,                # Stronger regularization
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    dt = DecisionTreeClassifier(
        max_depth=5,          # Reduce depth
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,   # Reduce learning rate
        max_depth=3,          # Reduce depth
        min_samples_split=10,
        subsample=0.8,        # Add subsampling
        random_state=42
    )

def train_models(X_train, X_test, y_train, y_test):
    global global_vectorizer
    
    # Modified TF-IDF parameters
    global_vectorizer = TfidfVectorizer(
        max_features=3000,    # Reduce features
        min_df=5,             # Minimum document frequency
        max_df=0.7,           # Maximum document frequency
        ngram_range=(1, 2),
        strip_accents='unicode',
        stop_words='english'
    )
    
    # Transform the text data
    X_train_vec = global_vectorizer.fit_transform(X_train)
    X_test_vec = global_vectorizer.transform(X_test)
    
    # Train and evaluate each model
    models = {
        'Logistic Regression': lr,
        'Decision Tree': dt,
        'Random Forest': rf,
        'Gradient Boosting': gb
    }
    
    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict
        y_pred = model.predict(X_test_vec)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
    
    return results

def predict_news(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Transform the text using the fitted vectorizer
    text_vec = global_vectorizer.transform([processed_text])
    
    # Get predictions from all models
    predictions = {
        'Logistic Regression': lr.predict(text_vec)[0],
        'Decision Tree': dt.predict(text_vec)[0],
        'Random Forest': rf.predict(text_vec)[0],
        'Gradient Boosting': gb.predict(text_vec)[0]
    }
    
    # Convert predictions to "True News" or "Fake News"
    results = {model: "True News" if pred == 1 else "Fake News" 
              for model, pred in predictions.items()}
    
    return results

# Main execution
if __name__ == "__main__":
    # Load and prepare data
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Initialize and train models
    initialize_models()
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Example usage
    while True:
        print("\nEnter news text to classify (or 'quit' to exit):")
        news_text = input()
        
        if news_text.lower() == 'quit':
            break
            
        predictions = predict_news(news_text)
        
        print("\nPredictions:")
        for model, prediction in predictions.items():
            print(f"{model}: {prediction}")