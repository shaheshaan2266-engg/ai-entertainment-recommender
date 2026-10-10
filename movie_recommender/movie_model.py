"""
Bollywood & Indian Movie Recommender Model.
Exposes a clean API for standalone movie recommendations and chaining.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender


class BollywoodMovieModel:
    def __init__(self):
        self.engine = UnifiedEntertainmentRecommender(auto_train=True)

    def recommend(self, movie_name: str, top_n: int = 10, era: str = None, genre: str = None) -> list[dict]:
        """Recommends movies similar to the given movie query."""
        results = self.engine.recommend(
            query=movie_name,
            mode="movies",
            era_filter=era,
            genre_filter=genre,
            top_n=top_n
        )
        return results.get("movies", [])


if __name__ == "__main__":
    model = BollywoodMovieModel()
    test_query = "Dangal"
    print(f"Top Movie Recommendations for '{test_query}':")
    recs = model.recommend(test_query, top_n=5)
    for i, m in enumerate(recs, 1):
        print(f"{i}. {m['title']} ({m.get('release_year')}) - Score: {m.get('similarity_score')} [{m.get('reason')}]")