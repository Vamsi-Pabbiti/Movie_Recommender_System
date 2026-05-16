# 🎬 Hybrid Movie Recommender System

A **Hybrid Movie Recommendation System** that suggests movies to users by combining **Content-Based Filtering** and **Collaborative Filtering**. This project leverages movie metadata and user ratings to provide accurate, personalized movie recommendations.

---

## 🚀 Features

- Hybrid recommendation combining:
  - **Content-Based Filtering:** Uses movie genres and features (TF-IDF + Cosine Similarity)
  - **Collaborative Filtering:** Uses user rating patterns (SVD Matrix Factorization)
- Adjustable **Alpha Slider** to control weight between content and collaborative scores
- Interactive **Streamlit Web App**
- Fast and modular production-ready code
- Supports **top-N recommendations** for any movie and user

---

## 🛠️ Technologies Used

- **Python 3.10**
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – TF-IDF & Cosine Similarity
- **Surprise** – Collaborative Filtering (SVD)
- **Streamlit** – Web app UI
- **Joblib** – Model persistence

---

## 📁 Project Structure
movie-recommender/
│
├── data/
│ ├── movies.csv
│ └── ratings.csv
│
├── models/
│ ├── svd_model.pkl
│ ├── content_similarity.pkl
│ └── movies.pkl
│
├── train.py # Trains and saves models
├── recommender.py # Hybrid recommendation logic
├── app.py # Streamlit web app
├── requirements.txt # Dependencies
└── README.md


---

## 🔹 Dataset

- **MovieLens 100k or latest dataset**  
  [MovieLens Datasets](https://grouplens.org/datasets/movielens/)
- Contains:
  - `movies.csv` → Movie metadata (movieId, title, genres)
  - `ratings.csv` → User ratings (userId, movieId, rating)

---

## ⚙️ How It Works

1. **Content-Based Filtering**
   - TF-IDF vectorization of movie genres
   - Cosine similarity between movies
2. **Collaborative Filtering**
   - SVD matrix factorization on user ratings
   - Predicts ratings for unseen movies
3. **Hybrid Recommendation**
   - Weighted combination of content and collaborative scores
   - Tunable weight via alpha slider
4. **Recommendation Output**
   - Top-N recommended movies displayed in Streamlit UI

---

