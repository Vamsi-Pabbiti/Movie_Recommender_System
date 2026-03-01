# recommender.py

import joblib
import numpy as np

# Load saved models
svd_model = joblib.load("models/svd_model.pkl")
content_similarity = joblib.load("models/content_similarity.pkl")
movies = joblib.load("models/movies.pkl")

movies = movies.reset_index(drop=True)

def hybrid_recommend(user_id, movie_title, n=10, alpha=0.5):

    if movie_title not in movies['title'].values:
        return []

    idx = movies[movies['title'] == movie_title].index[0]
    content_scores = list(enumerate(content_similarity[idx]))

    hybrid_scores = []

    for i, content_score in content_scores:
        movie_id = movies.iloc[i]['movieId']
        collab_score = svd_model.predict(user_id, movie_id).est

        final_score = (alpha * content_score) + ((1 - alpha) * collab_score)
        hybrid_scores.append((i, final_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    top_movies = hybrid_scores[1:n+1]

    results = []
    for i, _ in top_movies:
        results.append({
            "title": movies.iloc[i]['title'],
            "genres": movies.iloc[i]['genres']
        })

    return results