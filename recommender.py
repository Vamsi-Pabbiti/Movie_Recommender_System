# recommender.py

import joblib

# Load pre-trained models
svd_model = joblib.load("models/svd_model.pkl")
content_similarity = joblib.load("models/content_similarity.pkl")
movies = joblib.load("models/movies.pkl")

# Reset index for safe access
movies = movies.reset_index(drop=True)

def hybrid_recommend(user_id, movie_title, n=10, alpha=0.5):
    """
    Hybrid Movie Recommender
    
    Parameters:
    user_id : int - ID of the user
    movie_title : str - Movie selected by user
    n : int - Number of recommendations
    alpha : float - Weight for content-based score (0-1)
    
    Returns:
    List of dictionaries: [{"title": ..., "genres": ...}, ...]
    """
    if movie_title not in movies['title'].values:
        return []

    # Get index of the selected movie
    idx = movies[movies['title'] == movie_title].index[0]
    
    # Content-based similarity scores
    content_scores = list(enumerate(content_similarity[idx]))
    
    # Combine with collaborative score
    hybrid_scores = []
    for i, content_score in content_scores:
        movie_id = movies.iloc[i]['movieId']
        collab_score = svd_model.predict(user_id, movie_id).est
        
        final_score = (alpha * content_score) + ((1 - alpha) * collab_score)
        hybrid_scores.append((i, final_score))
    
    # Sort by final score descending
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N movies (skip the first, which is the same movie)
    top_movies = hybrid_scores[1:n+1]
    
    # Prepare results with title and genres
    results = []
    for i, _ in top_movies:
        results.append({
            "title": movies.iloc[i]['title'],
            "genres": movies.iloc[i]['genres']
        })
    
    return results