# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Fill missing values
movies['genres'] = movies['genres'].fillna('')

# Convert genres into text format
movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine Similarity Matrix
content_similarity = cosine_similarity(tfidf_matrix)

def hybrid_recommend(movie_title, n=10):

    # Check if movie exists
    if movie_title not in movies['title'].values:
        return []

    # Get movie index
    movie_idx = movies[movies['title'] == movie_title].index[0]

    # Similarity scores
    similarity_scores = list(enumerate(content_similarity[movie_idx]))

    # Sort movies by similarity
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Top recommendations
    top_movies = similarity_scores[1:n+1]

    # Return movie titles
    recommendations = [
        movies.iloc[i[0]]['title']
        for i in top_movies
    ]

    return recommendations