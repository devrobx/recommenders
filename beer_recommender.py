
from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import json

mlb = MultiLabelBinarizer()
model_svd = SVD()

l = []

with open('ratebeer.json') as f:
    for line in f:
        data = eval(line)
        l.append(data)
    df = pd.DataFrame(l)

user_ids = df["review/profileName"].unique().tolist()

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}

beer_ids = df["beer/beerId"].unique().tolist()

beer2beer_encoded = {x: i for i, x in enumerate(beer_ids)}
beer_encoded2beer = {i: x for i, x in enumerate(beer_ids)}

df["user"] = df["review/profileName"].map(user2user_encoded)
df["beer"] = df["beer/beerId"].map(beer2beer_encoded)

num_users = len(user2user_encoded)
num_beers = len(beer_encoded2beer)
df["rating"] = df["review/overall"].apply(lambda x: int(x.split("/")[0]) if type(x) != float else 0)

min_rating = min(df["rating"])
max_rating = max(df["rating"])

print("Number of users: {}, Number of Beers: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_beers, min_rating, max_rating))

# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# label encode user and beer
user_encoder = LabelEncoder() 
beer_encoder = LabelEncoder()

df["user"] = user_encoder.fit_transform(df["user"])
df["beer"] = beer_encoder.fit_transform(df["beer"])

# convert data to surprise objects 
train_df, test_df = train_test_split(df, test_size = 0.5)
reader = Reader(rating_scale = (0,1))
data = Dataset.load_from_df(train_df[['user', 'beer', 'rating']], reader)
trainset = data.build_full_trainset()

model_svd.fit(trainset)

predictions_svd = model_svd.test(trainset.build_anti_testset())
accuracy.rmse(predictions_svd)

def get_top_n_recommendations(user_id, n=5):
    user_liked_beers = df[df['user'] == user_id]['beer'].unique()
    beer = df['beer'].unique()

    beers_to_predict = list(set(beer) - set(user_liked_beers))
    
        # go through the movies and guess what the user would rate them
    user_beer_pairs = [(user,beer, 0) for beer in beers_to_predict]
    predictions_cf = model_svd.test(user_beer_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est, reverse = True)[:n]
    
    top_n_beer_ids = [int(pred.iid) for pred in top_n_recommendations]
    top_n_beer = beer_encoder.inverse_transform(top_n_beer_ids)

    return top_n_beer

# Reccomendations
user_id = df['review/profileName'].sample(1).iloc[0]

beers_rated_by_user = df[df['review/profileName'] == user_id]
n = 5

unique_beer_df = df[['beer/beerId', 'beer/style', 'beer/name', 'beer']].drop_duplicates()

print("High Rated Beers from user")
print("----" * 8)
top_beers_user = (beers_rated_by_user.sort_values(by="rating", ascending=False).head(5).beer.values)

beer_df_rows = unique_beer_df[unique_beer_df["beer"].isin(top_beers_user)]
for row in beer_df_rows.itertuples():
    print(row[3]+": "+row[2])

recommendations = get_top_n_recommendations(user_id)

print(f"Top {n} Recomendations for user {user}")
print(recommendations)
print("----" * 8)
recommended_beers = unique_beer_df[unique_beer_df["beer/beerId"].isin(recommended_beer_ids)]
for row in recommended_beers.itertuples():
    print(row[3]+": "+row[2])
    
