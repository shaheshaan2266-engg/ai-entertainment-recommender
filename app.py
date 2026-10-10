"""
Flask Web Application & REST API for AI Entertainment Recommender.
Provides fast endpoints for live search, hybrid recommendations, entity chaining, and curated sets.
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender

app = Flask(__name__)

# Initialize Recommender singleton
print("Loading AI Recommender Engine...")
recommender = UnifiedEntertainmentRecommender(auto_train=True)
print("AI Recommender Ready!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()
    try:
        limit = int(request.args.get("limit", 8))
    except (ValueError, TypeError):
        limit = 8
    results = recommender.search_autocomplete(query, limit=limit)
    return jsonify({"query": query, "results": results})


@app.route("/api/filter-options", methods=["GET"])
def filter_options():
    options = recommender.get_filter_options()
    return jsonify(options)


@app.route("/api/recommend", methods=["GET"])
def recommend():
    query = request.args.get("query", "").strip()
    mode = request.args.get("mode", "both").lower()
    era = request.args.get("era", "All")
    genre = request.args.get("genre", "All")
    genres = request.args.get("genres", "").strip()
    year = request.args.get("year", "All").strip()
    try:
        top_n = int(request.args.get("top_n", 12))
    except (ValueError, TypeError):
        top_n = 12

    if not query and not genres and year in ["All", "", None]:
        return jsonify({"error": "Query or filter parameters are required"}), 400

    results = recommender.recommend(
        query=query,
        mode=mode,
        era_filter=era,
        genre_filter=genre,
        genres_filter=genres,
        year_filter=year,
        top_n=top_n
    )
    return jsonify(results)


@app.route("/api/chain", methods=["GET"])
def chain():
    entity_type = request.args.get("type", "movie").lower()
    entity_name = request.args.get("name", "").strip()

    if not entity_name:
        return jsonify({"error": "Entity name is required"}), 400

    chain_data = recommender.graph.crawl_chain(entity_type=entity_type, entity_name=entity_name)
    return jsonify(chain_data)


@app.route("/api/trending", methods=["GET"])
def trending():
    data = recommender.get_curated_trending()
    return jsonify(data)


@app.route("/api/stats", methods=["GET"])
def stats():
    movies_count = len(recommender.df_movies)
    songs_count = len(recommender.df_songs)
    eras = recommender.df_movies['era'].value_counts().to_dict()
    return jsonify({
        "total_movies": movies_count,
        "total_songs": songs_count,
        "eras": eras
    })


@app.route("/api/watch-info", methods=["GET"])
def watch_info():
    title = request.args.get("title", "").strip()
    media_type = request.args.get("type", "movie").lower()

    if not title:
        return jsonify({"error": "Title parameter is required"}), 400

    from ai_entertainment_recommender.watch_provider import get_watch_info

    # Find metadata if available
    metadata = {}
    if media_type == "movie":
        matches = recommender.df_movies[recommender.df_movies['title'].str.lower() == title.lower()]
        if not matches.empty:
            metadata = matches.iloc[0].to_dict()
    else:
        matches = recommender.df_songs[recommender.df_songs['title'].str.lower() == title.lower()]
        if not matches.empty:
            metadata = matches.iloc[0].to_dict()

    info = get_watch_info(title=title, media_type=media_type, metadata=metadata)
    return jsonify(info)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
