# app.py

import streamlit as st
from recommender import hybrid_recommend
import joblib

movies = joblib.load("models/movies.pkl")

st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommender System")

st.sidebar.header("User Settings")

user_id = st.sidebar.number_input("Enter User ID", min_value=1, step=1)
alpha = st.sidebar.slider("Content Weight (Alpha)", 0.0, 1.0, 0.5)

selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Get Recommendations"):

    with st.spinner("Generating recommendations..."):
        recommendations = hybrid_recommend(user_id, selected_movie, n=10, alpha=alpha)

    st.subheader("Top Recommendations:")

    for movie in recommendations:
        st.markdown(f"⭐ **{movie['title']}**")
        st.markdown(f"🎭 *Genres:* {movie['genres']}")
        st.markdown("---")