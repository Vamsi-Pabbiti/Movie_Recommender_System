# recommender.py

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movies
movies = pd.read_csv("data/movies.csv")

# Clean genres
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Similarity matrix
content_similarity = cosine_similarity(tfidf_matrix)

def hybrid_recommend(movie_title, n=10):

    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]

    similarity_scores = list(enumerate(content_similarity[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_movies = similarity_scores[1:n+1]

    recommendations = [
        movies.iloc[i[0]]['title']
        for i in top_movies
    ]

    return recommendations