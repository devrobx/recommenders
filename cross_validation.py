from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import cross_validate
import pandas as pd

posts_df = pd.read_csv('posts_data.csv')

reader = Reader(rating_scale=(0, 1)) 
data = Dataset.load_from_df(posts_df[['user_id', 'item_id', 'liked']], reader)

algos = {
    "SVD": SVD(),
    "KNNBasic": KNNBasic(),
    "NMF": NMF()
}

# Perform cross-validation for each algorithm
for algo_name, algo in algos.items():
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print()
    print(f"Results for {algo_name}:")
    print("Average RMSE:", cv_results['test_rmse'].mean())
    print("Average MAE:", cv_results['test_mae'].mean())
    print()
