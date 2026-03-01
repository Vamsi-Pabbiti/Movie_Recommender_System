# train.py

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

print("Loading data...")

movies = pd.read_csv("C:\\Users\\Vamsi\\OneDrive\\Documents\\movie-recommender\\data\\movies.csv")
ratings = pd.read_csv("C:\\Users\\Vamsi\\OneDrive\\Documents\\movie-recommender\\data\\ratings.csv")

# -------------------------
# CONTENT-BASED MODEL
# -------------------------

print("Building Content-Based model...")

movies['genres'] = movies['genres'].str.replace('|', ', ')
movies['genres'] = movies['genres'].fillna('')

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

content_similarity = cosine_similarity(tfidf_matrix)

# -------------------------
# COLLABORATIVE MODEL
# -------------------------

print("Training Collaborative Filtering model...")

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.2)

svd_model = SVD(n_factors=100, n_epochs=20, random_state=42)
svd_model.fit(trainset)

# -------------------------
# SAVE MODELS
# -------------------------

print("Saving models...")

joblib.dump(svd_model, "models/svd_model.pkl")
joblib.dump(content_similarity, "models/content_similarity.pkl")
joblib.dump(movies, "models/movies.pkl")

print("Training completed successfully!")