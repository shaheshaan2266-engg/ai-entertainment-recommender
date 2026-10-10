"""
Bollywood & Indian Music Recommender Model.
Exposes a clean API for standalone song recommendations, singer chaining, and mood alignment.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender


class BollywoodMusicModel:
    def __init__(self):
        self.engine = UnifiedEntertainmentRecommender(auto_train=True)

    def recommend(self, song_name: str, top_n: int = 10, era: str = None) -> list[dict]:
        """Recommends songs similar to the given track or artist query."""
        results = self.engine.recommend(
            query=song_name,
            mode="songs",
            era_filter=era,
            top_n=top_n
        )
        return results.get("songs", [])


if __name__ == "__main__":
    model = BollywoodMusicModel()
    test_query = "Tum Hi Ho"
    print(f"Top Song Recommendations for '{test_query}':")
    recs = model.recommend(test_query, top_n=5)
    for i, s in enumerate(recs, 1):
        print(f"{i}. {s['title']} ({s.get('movie_title')}) - Singer: {s.get('singer')} [{s.get('reason')}]")
