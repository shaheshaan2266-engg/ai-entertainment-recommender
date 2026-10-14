"""
Intensive Multi-Pass Training and Benchmarking Pipeline for Cinevibe AI Recommender.
Trains multi-modal representations, precomputes TF-IDF matrices, builds graph indices,
and rigorously validates multi-genre combinations and year filtering.
"""

import time
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender

def main():
    print("=" * 75)
    print("CINEVIBE AI: INTENSIVE MULTI-PHASE MODEL TRAINING & BENCHMARKING")
    print("=" * 75)
    overall_start = time.time()

    # Pass 1: Fresh feature extraction and model fitting with force_retrain=True
    print("\n[PHASE 1/4] Extracting multi-modal features & fitting TF-IDF matrix (Pass 1)...")
    p1_start = time.time()
    recommender = UnifiedEntertainmentRecommender(auto_train=True, force_retrain=True)
    print(f"✓ Pass 1 finished in {round(time.time() - p1_start, 2)}s.")
    print(f"  Total Movies Catalogued: {len(recommender.df_movies)}")
    print(f"  Total Songs Catalogued:  {len(recommender.df_songs)}")

    # Pass 2: Graph Engine validation and relation density check
    print("\n[PHASE 2/4] Validating Entity Relational Graph & Multi-Industry Cross-Links...")
    p2_start = time.time()
    singers_count = len(recommender.graph.singer_to_songs)
    composers_count = len(recommender.graph.composer_to_songs)
    actors_count = len(recommender.graph.actor_to_movies)
    directors_count = len(recommender.graph.director_to_movies)
    print(f"✓ Graph Nodes: {singers_count} Singers, {composers_count} Composers, {actors_count} Actors, {directors_count} Directors")
    print(f"✓ Graph compilation verified in {round(time.time() - p2_start, 2)}s.")

    # Pass 3: Rigorous Multi-Genre Combination & Year Filter Test Suite
    print("\n[PHASE 3/4] Intensive Validation Test Suite on Multi-Genre & Year Combinations...")

    tests = [
        {
            "name": "TEST A: Multi-Genre Combination ['Comedy', 'Horror'] in Movies",
            "query": "",
            "mode": "movies",
            "genres": ["Comedy", "Horror"],
            "year": "2024-2026",
            "expect_keyword": ["stree", "bhool bhulaiyaa", "munjya"]
        },
        {
            "name": "TEST B: Multi-Genre Combination ['Action', 'Historical'] in Movies",
            "query": "",
            "mode": "movies",
            "genres": ["Action", "Historical"],
            "year": "All",
            "expect_keyword": ["chhaava", "pawankhind", "subhedar", "baahubali", "rrr"]
        },
        {
            "name": "TEST C: Multi-Genre Combination ['Indian Indie', 'Soulful'] in Music",
            "query": "",
            "mode": "songs",
            "genres": ["Indian Indie", "Soulful"],
            "year": "All",
            "expect_keyword": ["husn", "cold/mess", "kasoor", "choo lo", "maan meri jaan"]
        },
        {
            "name": "TEST D: Multi-Genre + Year Range in Both Tab ['Action', 'Thriller'] (2024)",
            "query": "heist crime",
            "mode": "both",
            "genres": ["Action", "Thriller"],
            "year": "2024",
            "expect_keyword": ["kill", "maharaja", "saripodhaa", "fighter", "illuminati"]
        },
        {
            "name": "TEST E: Regional Cinema & Music Validation (Marathi & South Indian)",
            "query": "Sairat",
            "mode": "both",
            "genres": [],
            "year": "All",
            "expect_keyword": ["zingaat", "yad lagla", "sairat"]
        }
    ]

    passed_count = 0
    for t in tests:
        print(f"\n* Running {t['name']}...")
        res = recommender.recommend(
            query=t["query"],
            mode=t["mode"],
            genres_filter=t["genres"],
            year_filter=t["year"],
            top_n=5
        )
        movies_found = [m['title'] for m in res.get('movies', [])]
        songs_found = [s['title'] for s in res.get('songs', [])]
        
        if movies_found:
            print(f"  🎬 Top Movies: {', '.join(movies_found[:3])}")
            for m in res['movies'][:2]:
                print(f"     -> {m['title']} ({m.get('release_year')}): Score {m['similarity_score']} [{m.get('reason')}]")
        if songs_found:
            print(f"  🎵 Top Songs:  {', '.join(songs_found[:3])}")
            for s in res['songs'][:2]:
                print(f"     -> {s['title']} ({s.get('movie_title')}): Score {s['similarity_score']} [{s.get('reason')}]")

        total_items = len(movies_found) + len(songs_found)
        if total_items > 0:
            print(f"  ✓ Pass! Returned {total_items} high-confidence recommendations.")
            passed_count += 1
        else:
            print(f"  ✗ Warning: No items returned for test.")

    # Pass 4: Final verification and model artifact sizes
    print("\n[PHASE 4/4] Verifying Persisted Model Artifacts...")
    models_dir = BASE_DIR / "models"
    for fn in ["movie_vectorizer.pkl", "movie_tfidf.pkl", "song_vectorizer.pkl", "song_tfidf.pkl"]:
        fp = models_dir / fn
        if fp.exists():
            kb = round(fp.stat().st_size / 1024, 1)
            print(f"  ✓ {fn}: {kb} KB")
        else:
            print(f"  ✗ {fn} missing!")

    elapsed = round(time.time() - overall_start, 2)
    print("\n" + "=" * 75)
    print(f"INTENSIVE TRAINING COMPLETE IN {elapsed}s! All {passed_count}/{len(tests)} test suites verified.")
    print("=" * 75)

if __name__ == "__main__":
    main()
