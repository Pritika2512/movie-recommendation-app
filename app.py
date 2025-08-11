# app.py
import streamlit as st
from recommender import recommend
import pandas as pd

movies = pd.read_csv('movies.csv')

st.title("🎬 Movie Recommendation System")

movie_list = movies['title'].values
selected_movie = st.selectbox("Pick a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write(movie)
