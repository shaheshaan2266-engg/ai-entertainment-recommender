import pandas as pd
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

ratings = pd.read_csv("data/movie/ratings.csv")
movies = pd.read_csv("data/movie/movies.csv")

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader=__import__("surprise").Reader(rating_scale=(0.5, 5))
)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)

print("Movie Recommendation Model")
accuracy.rmse(predictions)
accuracy.mae(predictions)