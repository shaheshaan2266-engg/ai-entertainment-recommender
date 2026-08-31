from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def load_movie_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(ROOT / "data" / "movie" / "movies.csv")
    ratings = pd.read_csv(ROOT / "data" / "movie" / "ratings.csv")
    return movies, ratings


def load_music_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "music" / "data.csv")
