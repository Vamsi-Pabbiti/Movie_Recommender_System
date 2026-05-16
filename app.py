# app.py

import streamlit as st
import pandas as pd
from recommender import hybrid_recommend

# Load dataset
movies = pd.read_csv("data/movies.csv")

# Page settings
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f5f5;
}

h1 {
    color: #222222;
    font-weight: 700;
}

.stButton>button {
    background-color: #ffffff;
    color: black;
    border-radius: 8px;
    border: 1px solid #cccccc;
    padding: 0.5rem 1rem;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #eeeeee;
}

.recommendation-card {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 0px 1px 4px rgba(0,0,0,0.1);
}

.genre-text {
    color: #666666;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("User Settings")

user_id = st.sidebar.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

alpha = st.sidebar.slider(
    "Content Weight (Alpha)",
    0.0,
    1.0,
    0.5
)

# Main title
st.title("🎬 Hybrid Movie Recommender System")

# Movie dropdown
selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

# Recommendation button
if st.button("Get Recommendations"):

    recommendations = hybrid_recommend(
        selected_movie,
        n=10
    )

    st.subheader("Top Recommendations:")

    for movie in recommendations:

        movie_data = movies[movies['title'] == movie].iloc[0]

        st.markdown(f"""
        <div class="recommendation-card">
            <h4>⭐ {movie_data['title']}</h4>
            <p class="genre-text">
                🎭 Genres: {movie_data['genres']}
            </p>
        </div>
        """, unsafe_allow_html=True)