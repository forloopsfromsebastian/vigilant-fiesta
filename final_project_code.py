#! usr/bin/env python3
"""
data can be downloaded at https://grouplens.org/datasets/movielens/
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
import random
from scipy.stats import pearsonr

# Load the datasets
movies = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest-small\movies.csv")
ratings = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest-small\ratings.csv")
genome_scores = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest\genome-scores.csv")
genome_tags = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest-small\tags.csv")


# Preprocess the data
movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))

# Extract the release year from the title and create a new 'year' column
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False).astype('float')


# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Prepare the ratings data for Surprise
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2)

# Train the SVD model
svd = SVD()
svd.fit(trainset)


def get_genre_preferences():
    genre_counts = movies['genres'].str.split(' ', expand=True).stack().value_counts()
    top_genres = genre_counts[:100].index.tolist()
    
    print("Available genres:")
    for genre in top_genres:
        print(genre)
    
    preferred_genre = input("\nEnter your preferred genre: ")
    return preferred_genre


def filter_data_by_genre(preferred_genre, min_year, max_year):
    # Filter movies by genre
    genre_movies = movies[movies['genres'].str.contains(preferred_genre)]

    # Filter movies by release year
    genre_movies = genre_movies[(genre_movies['year'] >= min_year) & (genre_movies['year'] <= max_year)]

    # Filter movies by rating threshold (example: minimum rating of 3)
    rating_threshold = 3.0
    genre_movies = genre_movies.merge(ratings.groupby('movieId')['rating'].mean().reset_index(),
                                      on='movieId', how='inner')
    genre_movies = genre_movies[genre_movies['rating'] >= rating_threshold]

    genre_ratings = ratings[ratings['movieId'].isin(genre_movies['movieId'])]
    
    return genre_movies, genre_ratings



def get_movie_to_rate(user_id):
    movie_ids = ratings['movieId'].unique()
    random.shuffle(movie_ids)

    for movie_id in movie_ids:
        if not ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == movie_id)].empty:
            continue

        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        return movie_title

    return None


def get_year_range():
    print("\nEnter your preferred range of movie years.")
    min_year = int(input("Enter the minimum year: "))
    max_year = int(input("Enter the maximum year: "))
    return min_year, max_year

def get_user_ratings(preferred_genre, min_year, max_year):
    genre_movies = movies[movies['genres'].str.contains(preferred_genre)]
    genre_movies = genre_movies[(genre_movies['year'] >= min_year) & (genre_movies['year'] <= max_year)]

    if genre_movies.empty:
        print("No movies found in the specified genre and year range.")
        return []

    avg_ratings = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    avg_ratings.reset_index(inplace=True)

    genre_movies = genre_movies.merge(avg_ratings, on='movieId')
    genre_movies = genre_movies.sort_values(by='rating', ascending=True)

    user_ratings = []

    while len(user_ratings) < 3:
        for _, movie in genre_movies.iterrows():
            title = movie['title']
            movie_id = movie['movieId']
            print(f"\nMovie to rate: {title}")
            seen_movie = input("Have you seen this movie? (yes or no): ")

            if seen_movie.lower() == "yes":
                rating = float(input("Enter your rating for this movie (0.5 to 5.0): "))
                user_ratings.append((movie_id, rating))

                if len(user_ratings) >= 3:
                    break

    return user_ratings


    while len(user_ratings) < 3:
        for _, movie in genre_movies.iterrows():
            title = movie['title']
            movie_id = movie['movieId']
            print(f"\nMovie to rate: {title}")
            seen_movie = input("Have you seen this movie? (yes or no): ")

            if seen_movie.lower() == "yes":
                rating = float(input("Enter your rating for this movie (0.5 to 5.0): "))
                user_ratings.append((movie_id, rating))

                if len(user_ratings) >= 3:
                    break

    return user_ratings



def assign_user_id(user_ratings):
    max_corr = -1
    assigned_user_id = -1
    movie_ids = [movie_id for movie_id, _ in user_ratings]
    
    # Keep only users who have rated at least one movie from user_ratings
    filtered_ratings = ratings[ratings['movieId'].isin(movie_ids)]
    
    for uid in filtered_ratings['userId'].unique():
        user_ratings_data = filtered_ratings[filtered_ratings['userId'] == uid][['movieId', 'rating']].set_index('movieId')
        new_user_ratings_data = pd.DataFrame(user_ratings, columns=['movieId', 'rating']).set_index('movieId')
        
        # Keep only movies rated both by the current user and the new user
        common_movie_ids = user_ratings_data.index.intersection(new_user_ratings_data.index)
        if not common_movie_ids.empty:
            user_ratings_data = user_ratings_data.loc[common_movie_ids]
            new_user_ratings_data = new_user_ratings_data.loc[common_movie_ids]

        if len(user_ratings_data) > 1 and len(new_user_ratings_data) > 1:
            corr, _ = pearsonr(user_ratings_data['rating'], new_user_ratings_data['rating'])
            if corr > max_corr:
                max_corr = corr
                assigned_user_id = uid


        return assigned_user_id




def hybrid_recommendations(user_id, user_ratings):
    estimated_ratings = []

    for movie_id, _ in user_ratings:
        similar_movies = get_similar_movie(movie_id, k=50)
        
        for movie_idx in similar_movies:
            movie_id = movies.iloc[movie_idx]['movieId']
            estimated_rating = svd.predict(user_id, movie_id).est
            estimated_ratings.append((movie_idx, estimated_rating))

    estimated_ratings = sorted(estimated_ratings, key=lambda x: x[1], reverse=True)

    recommended_movies = [movies.iloc[i[0]]['title'] for i in estimated_ratings[:10]]

    return recommended_movies

def get_similar_movie(movie_id, k=50):
    idx = movies[movies['movieId'] == movie_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_similar_movies = [i[0] for i in sim_scores[1:k+1]]
    return top_similar_movies


def main():
    # Load the datasets
    movies = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest-small\movies.csv")
    ratings = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-latest-small\ratings.csv")
    genome_scores = pd.read_csv(r"C:\Users\sebsl\OneDrive\Documents\COMP3006spring\ml-25m\genome-scores.csv")

    # Preprocess the data
    movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))
    # Extract the release year from the title and create a new 'year' column
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)', expand=False).astype('float')

    # Compute TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies['genres'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Prepare the ratings data for Surprise
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train the SVD model
    svd = SVD()
    svd.fit(trainset)
    
    preferred_genre = get_genre_preferences()
    min_year, max_year = get_year_range()
    user_ratings = get_user_ratings(preferred_genre, min_year, max_year)
    assigned_user_id = assign_user_id(user_ratings)
    print(f"\nYour assigned user ID is {assigned_user_id}.")

    recommendations = hybrid_recommendations(assigned_user_id, user_ratings)
    print("\nTop 10 recommended movies:")
    for i, title in enumerate(recommendations, start=1):
        print(f"{i}. {title}")

    
if __name__ == "__main__":
    main()