import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD


RATINGS_PATH = '/Users/ahalyaanil/Documents/Python_projects/ml-32m/ratings.csv'
MOVIES_PATH = '/Users/ahalyaanil/Documents/Python_projects/ml-32m/movies.csv'
USER_RATINGS_PATH = 'user_ratings.csv'  


ratings_df = pd.read_csv(RATINGS_PATH)
movies_df = pd.read_csv(MOVIES_PATH)

movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)')[0].astype(float)
movies_df = movies_df[movies_df['year'] >= 2000]


ratings_df = ratings_df.sample(n=100000, random_state=42)

try:
    user_ratings_df = pd.read_csv(USER_RATINGS_PATH)
except FileNotFoundError:
    user_ratings_df = pd.DataFrame(columns=['userId','movieId','rating'])

st.title("🎬 Movie Recommender System")


user_id = st.text_input("Enter your User ID:", value="7000")

st.write("Rate the following movies:")


if 'movie_options' not in st.session_state:
    st.session_state.movie_options = movies_df.sample(5)

movie_options = st.session_state.movie_options
user_ratings = []

for idx, row in movie_options.iterrows():
    rating = st.slider(f"{row['title']}:", 1, 5, 3)
    user_ratings.append((int(user_id), int(row['movieId']), rating))

if st.button("Get Recommendations"):

   
    new_ratings_df = pd.DataFrame(user_ratings, columns=['userId','movieId','rating'])
    user_ratings_df = pd.concat([user_ratings_df, new_ratings_df])
    user_ratings_df.drop_duplicates(subset=['userId','movieId'], keep='last', inplace=True)
    user_ratings_df.to_csv(USER_RATINGS_PATH, index=False)

    st.session_state.movie_options = movies_df.sample(5)

    combined_df = pd.concat([ratings_df[['userId','movieId','rating']], user_ratings_df])

    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(combined_df[['userId','movieId','rating']], reader)
    trainset = data.build_full_trainset()

    algo = SVD()
    algo.fit(trainset)

    all_movie_ids = set(movies_df['movieId'])
    rated_movie_ids = set(user_ratings_df[user_ratings_df['userId'] == int(user_id)]['movieId'])
    unrated_movie_ids = list(all_movie_ids - rated_movie_ids)

    predictions = [(iid, algo.predict(int(user_id), iid).est) for iid in unrated_movie_ids]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_5 = predictions[:5]

    st.subheader(f"Top 5 recommended movies for User {user_id}:")
    for iid, rating in top_5:
        title = movies_df[movies_df['movieId'] == iid]['title'].values[0]
        st.write(f"{title} - predicted rating: {rating:.2f}")
