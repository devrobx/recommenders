from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# handles multi label categorical data - makes it useable for machine learning
mlb = MultiLabelBinarizer()
model_svd = SVD()

# read data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# combine data
df = pd.merge(ratings, movies[["movieId", "genres"]], on = "movieId", how="left")

# label encode user and movies
user_encoder = LabelEncoder() 
movie_encoder = LabelEncoder()

# help learn the data before processing
df["userId"] = user_encoder.fit_transform(df["userId"])
df["movieId"] = movie_encoder.fit_transform(df["movieId"])

# split genres into single columns of true and false
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('genres').str.split('|')), columns = mlb.classes_,index = df.index))

df.drop(columns = "(no genres listed)", inplace = True)

# convert data to surprise objects 

train_df, test_df = train_test_split(df, test_size = 0.2)
reader = Reader(rating_scale = (0.5,5))
data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

model_svd.fit(trainset)

predictions_svd = model_svd.test(trainset.build_anti_testset())
accuracy.rmse(predictions_svd)

# Make Recommendation
def get_top_n_recommendations(user_id, n=10):
    user_movies = df[df['userId'] == user_id]['movieId'].unique()
    all_movies = df['movieId'].unique()
    movies_to_predict = list(set(all_movies) - set(user_movies))
    
    # go through the movies and guess what the user would rate them
    user_movie_pairs = [(user_id, movie_id, 0) for movie_id in movies_to_predict]
    predictions_cf = model_svd.test(user_movie_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est, reverse = True)[:n]
    
    top_n_movie_ids = [int(pred.iid) for pred in top_n_recommendations]

    # convert labels to their original categorical form to interpret machine learning model
    top_n_movies = movie_encoder.inverse_transform(top_n_movie_ids)

    return top_n_movies

# my ratings
user_id = 611
n = 10
recommendations = get_top_n_recommendations(user_id, n)
movie_recommendations_by_title = movies[movies['movieId'].isin(recommendations)]['title'].tolist()

print(f"Top {n} Recomendations for user {user_id}):")

for i, title in enumerate(movie_recommendations_by_title, 1):
    print(f"{i}.{title}")

