from surprise import Dataset, Reader
from surprise import SVD
from surprise import accuracy
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split


mlb = MultiLabelBinarizer()
model_svd = SVD()


users_df = pd.read_csv('users_data.csv')
posts_df = pd.read_csv('posts_data.csv')

df = pd.merge(posts_df, users_df, on='user_id', how='left')
print("Merged data")
# label encode user and movies
user_encoder = LabelEncoder() 
post_encoder = LabelEncoder()

df["user_id"] = user_encoder.fit_transform(df["user_id"])
df["item_id"] = post_encoder.fit_transform(df["item_id"])

# convert data to surprise objects 
train_df, test_df = train_test_split(df, test_size = 0.2)
reader = Reader(rating_scale = (0,1))
data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'liked']], reader)
trainset = data.build_full_trainset()

print("trained data")

model_svd.fit(trainset)

predictions_svd = model_svd.test(trainset.build_anti_testset())
accuracy.rmse(predictions_svd)

def get_top_n_recommendations(user_id, n=5):
    user_liked_posts = df[df['user_id'] == user_id]['item_id'].unique()
    posts = df['item_id'].unique()

    posts_to_predict = list(set(posts) - set(user_liked_posts))
    
        # go through the movies and guess what the user would rate them
    user_posts_pairs = [(user_id,item_id, 0) for item_id in posts_to_predict]
    predictions_cf = model_svd.test(user_posts_pairs)

    top_n_recommendations = sorted(predictions_cf, key = lambda x: x.est, reverse = True)[:n]
    
    top_n_post_ids = [int(pred.iid) for pred in top_n_recommendations]
    top_n_posts = post_encoder.inverse_transform(top_n_post_ids)

    return top_n_posts

def get_username(user_id):
    return df.loc[df['user_id'] == user_id, 'user_name'].iloc[0]


# Reccomendations
user_id = 10
user_name = get_username(user_id)
n = 5
recommendations = get_top_n_recommendations(user_id)

post_recommendations_by_title = posts_df[posts_df['item_id'].isin(recommendations)]['title'].tolist()

print(f"Top {n} Recomendations for user {user_name}")

for i, title in enumerate(post_recommendations_by_title, 1):
    print(f"{i}.{title}")

    
