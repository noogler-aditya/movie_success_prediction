
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import ast

# Load the dataset
try:
    df = pd.read_csv('Downloads/tmdb_5000_movies.csv')  # Ensure the file is in the correct path
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File 'tmdb_5000_movies.csv' not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Clean and preprocess the data
def clean_data(df):
    # Filter out rows with invalid budget or revenue
    df = df[(df['budget'] > 0) & (df['revenue'] > 0)].copy()
    
    # Calculate ROI and create target variable 'success'
    df['ROI'] = (df['revenue'] - df['budget']) / df['budget']
    df['success'] = (df['ROI'] > 2) & (df['vote_average'] > 7.0)
    
    # Extract primary genre
    df['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notna(x) else [])
    df['primary_genre'] = df['genres'].apply(lambda x: x[0] if x else 'Unknown')
    
    # Extract director popularity (if 'crew' column exists)
    if 'crew' in df.columns:
        df['crew'] = df['crew'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        df['director_popularity'] = df['crew'].apply(
            lambda x: next((m['popularity'] for m in x if m.get('job') == 'Director'), 0)
        )
    else:
        df['director_popularity'] = 0  # Default value if 'crew' column is missing

    if 'cast' in df.columns:
        df['cast'] = df['cast'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
        df['cast_popularity'] = df['cast'].apply(
            lambda x: next((m['popularity'] for m in x), 0)  # Use first cast member's popularity
        )
    else:
        df['cast_popularity'] = 0  # Default value if 'cast' column is missing
    
        
    return df
        
# Clean the dataset
df = clean_data(df)

# Define features and target
features = ['budget', 'primary_genre', 'director_popularity', 'cast_popularity']
X = df[features]
y = df['success']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing numeric values
        ('scaler', StandardScaler())  # Scale numeric features
    ]), ['budget', 'director_popularity', 'cast_popularity']),  # Added cast_popularity
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['primary_genre']) 
])

# Create and train the model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the model
joblib.dump(model, 'movie_success_model.pkl')
print("Model trained and saved successfully!")