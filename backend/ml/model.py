import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
# UPDATE THIS PATH TO MATCH YOUR ACTUAL FILE LOCATION
DATASET_PATH = "D:/assisflow/dataset/Aa_dataset-Tickets-Multi-lang-5-2-50-version.csv" 
MODEL_DIR = "models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- PREPROCESSING ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    return text.strip()

print("1. Loading and Cleaning Data...")
try:
    df = pd.read_csv(DATASET_PATH)
    df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
    df['text'] = df['text'].apply(clean_text)
    
    # Priority Cleanup
    if 'priority' in df.columns:
        df['priority'] = df['priority'].str.capitalize().replace({'Critical': 'High'})
        df = df[df['priority'].isin(['Low', 'Medium', 'High'])]
    else:
        # Fallback for testing if column is missing
        df['priority'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))

    # Sentiment Cleanup (Create if missing)
    if 'sentiment' not in df.columns:
        df['sentiment'] = np.random.choice(['Positive', 'Negative', 'Neutral'], size=len(df))
        
except Exception as e:
    print(f"ERROR: Could not load data from {DATASET_PATH}")
    print(f"Details: {e}")
    exit()

# --- TRAIN PRIORITY MODEL (The Fix) ---
print("\n2. Training Priority Model...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['priority'], test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Using GridSearch (Optimized Training)
param_grid = {
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [None, 20]
}
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

# *** CRITICAL FIX: Get the best TRAINED model ***
best_priority_model = grid.best_estimator_ 

print(f"   - Training Complete. Accuracy: {best_priority_model.score(X_test, y_test):.2f}")

# --- TRAIN SENTIMENT MODEL ---
print("\n3. Training Sentiment Model...")
y_sent = df['sentiment']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(df['text'], y_sent, test_size=0.2, random_state=42)

sentiment_model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('clf', LogisticRegression(max_iter=1000))
])
sentiment_model.fit(X_train_s, y_train_s)
print(f"   - Training Complete. Accuracy: {sentiment_model.score(X_test_s, y_test_s):.2f}")

# --- SAVE CORRECTLY ---
print("\n4. Saving Models...")
# We save 'best_priority_model', NOT the raw pipeline
joblib.dump(best_priority_model, f'{MODEL_DIR}priority_pipeline.pkl')
joblib.dump(sentiment_model, f'{MODEL_DIR}sentiment_pipeline.pkl')

print("SUCCESS: Models have been re-trained and saved correctly.")











# import pandas as pd
# import numpy as np
# import re
# import joblib
# import os
# import seaborn as sns
# import matplotlib.pyplot as plt



# from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold , GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
# from textblob import TextBlob 

# # --- 1. CONFIGURATION ---
# DATASET_PATH = "D:/assisflow/dataset/Aa_dataset-Tickets-Multi-lang-5-2-50-version.csv" 
# MODEL_DIR = "models/"
# METRICS_DIR = "metrics/" # Folder to save confusion matrix images
# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(METRICS_DIR, exist_ok=True)

# # --- 2. PREPROCESSING FUNCTIONS ---
# def clean_text(text):
#     """
#     Basic cleaning: lowercase, remove special chars, strip whitespace.
#     """
#     if not isinstance(text, str):
#         return ""
#     text = text.lower()
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
#     return text.strip()

# # --- 3. LOAD & PREPARE DATA ---
# print("Loading dataset...")
# try:
#     df = pd.read_csv(DATASET_PATH)
    
#     # Combine subject and description
#     df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
#     df['text'] = df['text'].apply(clean_text)
    
#     # --- FILTER PRIORITIES (Low, Medium, High) ---
#     if 'priority' in df.columns:
#         df['priority'] = df['priority'].str.capitalize()
        
#         # Map Critical -> High, then Filter
#         df['priority'] = df['priority'].replace({'Critical': 'High'})
#         valid_priorities = ['Low', 'Medium', 'High']
#         df = df[df['priority'].isin(valid_priorities)]
        
#         print(f"Filtered Data Shape: {df.shape}")
#         print("Class Distribution:\n", df['priority'].value_counts())
#     else:
#         print("Warning: 'priority' column not found. Creating dummy data.")
#         df['priority'] = np.random.choice(['Low', 'Medium', 'High'], size=len(df))

# except Exception as e:
#     print(f"Error loading data: {e}")
#     exit()

# # --- 4. PRIORITY MODEL: TRAINING & CROSS-VALIDATION ---
# print("\n" + "="*40)
# print("   TRAINING PRIORITY MODEL")
# print("="*40)

# X = df['text']
# y_priority = df['priority']

# # Split Data
# X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.2, random_state=42, stratify=y_priority)

# # Define Pipeline
# priority_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))),
#     ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
# ])

# # --- CROSS VALIDATION STEP ---

# param_grid = {
#     'tfidf__max_features': [3000, 5000],          # Try smaller and larger vocabularies
#     'tfidf__ngram_range': [(1, 1), (1, 2)],       # Try single words vs phrases
#     'clf__n_estimators': [100, 200],              # Try different numbers of trees
#     'clf__max_depth': [None, 20],                 # Prevent overfitting
# }

# print("Starting Grid Search with Cross-Validation...")
# print("This will train multiple models to find the absolute best one.")

# # GridSearchCV automatically performs Cross Validation on every combination
# grid_search = GridSearchCV(
#     priority_pipeline, 
#     param_grid, 
#     cv=3,                 # 3-Fold Cross Validation
#     scoring='f1_weighted', # Optimize for F1 Score (better for imbalanced classes)
#     n_jobs=-1,            # Use all CPU cores
#     verbose=1
# )

# grid_search.fit(X_train, y_train)

# # --- GET THE BEST OPTIMIZED MODEL ---
# best_model = grid_search.best_estimator_

# print(f"\nBest Parameters Found: {grid_search.best_params_}")
# print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# # --- 5. EVALUATION ---
# print("\n" + "-"*30)
# print("   TESTING OPTIMIZED MODEL")
# print("-"*30)

# y_pred = best_model.predict(X_test)

# print("Classification Report:")
# print(classification_report(y_test, y_pred))



# ####
# # print("Running 5-Fold Cross-Validation (this may take a moment)...")
# # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# # cv_scores = cross_val_score(priority_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')

# # print(f"Cross-Validation Accuracy Scores: {cv_scores}")
# # print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# # # Fit on full training data
# # print("Training final model on full training set...")
# # priority_pipeline.fit(X_train, y_train)

# # # --- 5. PRIORITY MODEL: METRICS & EVALUATION ---
# # print("\n" + "-"*30)
# # print("   TESTING METRICS (Priority)")
# # print("-"*30)

# # y_pred = priority_pipeline.predict(X_test)

# # 1. Accuracy
# acc = accuracy_score(y_test, y_pred)
# print(f"Test Set Accuracy: {acc:.4f}")

# # 2. F1 Score (Weighted)
# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"Weighted F1 Score: {f1:.4f}")

# # 3. Detailed Report
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # 4. Confusion Matrix (Saved as Image)
# cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
# plt.figure(figsize=(8,6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
# plt.title('Confusion Matrix - Priority Model')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.savefig(f'{METRICS_DIR}priority_confusion_matrix.png')
# print(f"Confusion Matrix saved to {METRICS_DIR}priority_confusion_matrix.png")


# # --- 6. SENTIMENT MODEL (Brief) ---
# print("\n" + "="*40)
# print("   TRAINING SENTIMENT MODEL")
# print("="*40)

# # Generate Silver Labels if needed
# def get_sentiment_label(text):
#     score = TextBlob(text).sentiment.polarity
#     if score > 0.05: return 'Positive' 
#     if score < -0.05: return 'Negative'
#     return 'Neutral'

# if 'sentiment' not in df.columns:
#     df['sentiment'] = df['text'].apply(get_sentiment_label)

# y_sentiment = df['sentiment']
# X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sentiment, test_size=0.2, random_state=42)

# sentiment_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
#     ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
# ])

# # CV for Sentiment
# cv_scores_s = cross_val_score(sentiment_pipeline, X_train_s, y_train_s, cv=3, scoring='accuracy')
# print(f"Sentiment Mean CV Accuracy: {cv_scores_s.mean():.4f}")

# sentiment_pipeline.fit(X_train_s, y_train_s)
# print("Sentiment Model Trained.")

# # --- 7. SAVE ARTIFACTS ---
# print("\nSaving Models...")
# joblib.dump(priority_pipeline, f'{MODEL_DIR}priority_pipeline.pkl')
# joblib.dump(sentiment_pipeline, f'{MODEL_DIR}sentiment_pipeline.pkl')
# print("All models saved successfully.")