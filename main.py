from ai_entertainment_recommender.data_loader import load_movie_data, load_music_data


def main() -> int:
    movies, ratings = load_movie_data()
    music = load_music_data()

    print("Movie dataset sample:")
    print(movies.head())
    print(f"\nMovies rows: {len(movies)}")
    print(f"Ratings rows: {len(ratings)}")

    print("\nMusic dataset sample:")
    print(music.head())
    print(f"\nMusic rows: {len(music)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
