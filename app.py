import streamlit as st
import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#st.set_page_config(page_title="Movie Recommendation System")
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


#st.title("🎬 Movie Recommendation System")
st.title("🎬 Movie Recommendation System")

st.markdown(
    "Find movies similar to your favourite film using **Content-Based Recommendation** powered by **TF-IDF** and **Cosine Similarity**."
)
# sidebar
with st.sidebar:

    st.header("About")

    st.info(
        """
        This recommendation system suggests movies based on their content.

        It compares:
        - Genres
        - Cast
        - Director
        - Keywords
        - Tagline

        Recommendation Engine:
        • TF-IDF Vectorization
        • Cosine Similarity
        """
    )





movies_data = pd.read_csv("movies.csv")

selected_features = [
    "genres",
    "keywords",
    "tagline",
    "cast",
    "director"
]

for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna("")

combined_features = (
    movies_data["genres"] + " " +
    movies_data["keywords"] + " " +
    movies_data["tagline"] + " " +
    movies_data["cast"] + " " +
    movies_data["director"]
)

vectorizer = TfidfVectorizer()

feature_vectors = vectorizer.fit_transform(combined_features)

similarity = cosine_similarity(feature_vectors)

movie_list = sorted(movies_data["title"].tolist())

movie_name = st.selectbox(
    "🎬 Search & Select a Movie",
    movie_list,
    index=None,
    placeholder="Start typing a movie name..."
)

recommend = st.button("🎯 Recommend Similar Movies")

if recommend:

  if recommend:

    if movie_name is None:
        st.warning("Please select a movie.")
    else:

        close_match = movie_name

        index_of_the_movie = movies_data[
            movies_data.title == close_match
        ]["index"].values[0]

        similarity_score = list(
            enumerate(similarity[index_of_the_movie])
        )

        sorted_similar_movies = sorted(
            similarity_score,
            key=lambda x: x[1],
            reverse=True
        )

        st.success(f"🎬 Because you liked **{close_match}**, you may also enjoy:")

        recommendations = []

        for movie in sorted_similar_movies:

            index = movie[0]

            title = movies_data.loc[index, "title"]

            if title != close_match:
                recommendations.append(title)

        recommendations = recommendations[:12]

        cols = st.columns(3)

        for i, movie in enumerate(recommendations):

            with cols[i % 3]:
                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #ddd;
                        border-radius:12px;
                        padding:15px;
                        margin-bottom:15px;
                        text-align:center;
                        background:#ffffff;
box-shadow:0px 2px 8px rgba(0,0,0,0.15);
                        color:black;
                        font-weight:bold;
                    ">
                    🎥<br><br>
                    {movie}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")

st.caption(
    "Developed by Rav | Movie Recommendation System using TF-IDF, Cosine Similarity and Streamlit"
)