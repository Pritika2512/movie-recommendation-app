
# recommender.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('movies.csv')  # Ensure this file exists in the same folder

# Fill missing genres with an empty string
movies['genres'] = movies['genres'].fillna('')

# Convert genres to vectors
cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform(movies['genres'])

# Compute similarity for a given index
def compute_similarity(idx):
    movie_vec = vectors[idx]
    similarity_scores = cosine_similarity(movie_vec, vectors).flatten()
    return similarity_scores

# Recommendation function
def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return ["Movie not found in database."]
    
    idx = movies[movies['title'] == movie_title].index[0]
    similarity_scores = compute_similarity(idx)
    
    sim_scores = list(enumerate(similarity_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    recommended_movies = [movies.iloc[i[0]].title for i in sim_scores[1:6]]
    return recommended_movies
