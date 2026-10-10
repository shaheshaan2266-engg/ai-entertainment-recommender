from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def load_bollywood_movies() -> pd.DataFrame:
    path = ROOT / "data" / "bollywood" / "bollywood_movies.csv"
    if not path.exists():
        raise FileNotFoundError(f"Bollywood movies dataset not found at {path}. Run scripts/build_bollywood_dataset.py first.")
    return pd.read_csv(path)


def load_bollywood_songs() -> pd.DataFrame:
    path = ROOT / "data" / "bollywood" / "bollywood_songs.csv"
    if not path.exists():
        raise FileNotFoundError(f"Bollywood songs dataset not found at {path}. Run scripts/build_bollywood_dataset.py first.")
    return pd.read_csv(path)


def load_movie_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.read_csv(ROOT / "data" / "movie" / "movies.csv")
    ratings = pd.read_csv(ROOT / "data" / "movie" / "ratings.csv")
    return movies, ratings


def load_music_data() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "music" / "data.csv")
