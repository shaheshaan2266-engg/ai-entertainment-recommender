import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ai_entertainment_recommender.recommender import UnifiedEntertainmentRecommender
from ai_entertainment_recommender.database import (
    init_db, create_user, authenticate_user, get_user_by_id, update_user_profile,
    get_user_saved_items, add_user_saved_item, remove_user_saved_item, clear_user_saved_items,
    get_user_ratings, set_user_rating, ensure_default_user
)
from ai_entertainment_recommender.quick_explore import (
    get_top_10_movies, get_top_10_songs, get_trending_artists, get_latest_releases
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "cinevibe-ai-cinema-intelligence-secret-2026")

# Initialize database tables and seed Shiva profile if needed
init_db()

# Initialize Recommender singleton
print("Loading AI Recommender Engine...")
recommender = UnifiedEntertainmentRecommender(auto_train=True)
print("AI Recommender Ready!")


@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------------------------------------
# AUTHENTICATION & PROFILE ENDPOINTS
# -------------------------------------------------------------

def get_current_user_from_session():
    user_id = session.get("user_id")
    if user_id:
        return get_user_by_id(user_id)
    return None


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    user = get_current_user_from_session()
    if user:
        # Don't send password hash to client
        user_safe = {k: v for k, v in user.items() if k != "password_hash"}
        return jsonify({"authenticated": True, "user": user_safe})
    return jsonify({"authenticated": False, "user": None})


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.get_json(force=True, silent=True) or {}
    display_name = str(data.get("displayName") or "").strip()
    email = str(data.get("email") or "").strip().lower()
    password = str(data.get("password") or "")
    confirm_password = str(data.get("confirmPassword") or "")
    avatar = str(data.get("avatar") or "🎬")

    if not email or not password:
        return jsonify({"error": "Email/Username and password are required."}), 400

    if confirm_password and password != confirm_password:
        return jsonify({"error": "Passwords do not match."}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters long."}), 400

    try:
        user = create_user(
            display_name=display_name or email.split("@")[0].title(),
            email_or_username=email,
            password=password,
            avatar=avatar
        )
        session["user_id"] = user["id"]
        user_safe = {k: v for k, v in user.items() if k != "password_hash"}
        return jsonify({"success": True, "user": user_safe})
    except Exception as e:
        err_msg = str(e)
        if "UNIQUE" in err_msg.upper():
            return jsonify({"error": "An account with this email/username already exists."}), 409
        return jsonify({"error": f"Registration failed: {err_msg}"}), 500


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(force=True, silent=True) or {}
    email_or_username = str(data.get("email") or data.get("username") or "").strip().lower()
    password = str(data.get("password") or "")

    if not email_or_username or not password:
        return jsonify({"error": "Email/Username and password are required."}), 400

    user = authenticate_user(email_or_username, password)
    if not user:
        return jsonify({"error": "Invalid credentials. Please verify your email and password."}), 401

    session["user_id"] = user["id"]
    user_safe = {k: v for k, v in user.items() if k != "password_hash"}
    return jsonify({"success": True, "user": user_safe})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user_id", None)
    return jsonify({"success": True})


@app.route("/api/auth/sync-local", methods=["POST"])
def auth_sync_local():
    """Safely links & merges browser-local saved items and ratings into authenticated user account."""
    user = get_current_user_from_session()
    if not user:
        # Auto-fallback to default user Shiva if not explicitly authenticated
        user = ensure_default_user()
        session["user_id"] = user["id"]

    data = request.get_json(force=True, silent=True) or {}
    my_list = data.get("myList") or []
    favorites = data.get("favorites") or []
    ratings = data.get("ratings") or {}

    uid = user["id"]
    for item in my_list:
        add_user_saved_item(
            user_id=uid,
            title=item.get("title", ""),
            item_type=item.get("type", "movie"),
            list_type="list",
            sub=item.get("sub", "")
        )

    for item in favorites:
        add_user_saved_item(
            user_id=uid,
            title=item.get("title", ""),
            item_type=item.get("type", "movie"),
            list_type="favorites",
            sub=item.get("sub", "")
        )

    for key, tier in ratings.items():
        if ":" in key:
            t, mtype = key.split(":", 1)
            set_user_rating(uid, t, mtype, tier)

    return jsonify({"success": True, "user": {k: v for k, v in user.items() if k != "password_hash"}})


@app.route("/api/profile/update", methods=["POST"])
def profile_update():
    user = get_current_user_from_session()
    if not user:
        # Default user Shiva fallback
        user = ensure_default_user()
        session["user_id"] = user["id"]

    data = request.get_json(force=True, silent=True) or {}
    name = str(data.get("name") or user["display_name"]).strip()
    avatar = str(data.get("avatar") or user["avatar"]).strip()

    updated = update_user_profile(user["id"], name, avatar)
    user_safe = {k: v for k, v in updated.items() if k != "password_hash"}
    return jsonify({"success": True, "user": user_safe})


# -------------------------------------------------------------
# USER-SPECIFIC SAVED ITEMS (MY LIST & FAVORITES)
# -------------------------------------------------------------

@app.route("/api/user/saved", methods=["GET", "POST", "DELETE"])
def user_saved():
    user = get_current_user_from_session()
    if not user:
        return jsonify({"error": "Authentication required to access saved items.", "auth_required": True}), 401

    uid = user["id"]

    if request.method == "GET":
        list_type = request.args.get("list_type")  # 'list' or 'favorites' or None
        items = get_user_saved_items(uid, list_type=list_type)
        return jsonify({"items": items})

    elif request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        title = data.get("title")
        item_type = data.get("type", "movie")
        list_type = data.get("list_type", "list")
        sub = data.get("sub", "")
        if not title:
            return jsonify({"error": "Title is required"}), 400
        add_user_saved_item(uid, title, item_type, list_type, sub=sub)
        return jsonify({"success": True, "saved": True})

    elif request.method == "DELETE":
        data = request.get_json(force=True, silent=True) or {}
        title = data.get("title") or request.args.get("title")
        item_type = data.get("type", "movie") or request.args.get("type", "movie")
        list_type = data.get("list_type", "list") or request.args.get("list_type", "list")
        if not title:
            return jsonify({"error": "Title is required"}), 400
        remove_user_saved_item(uid, title, item_type, list_type)
        return jsonify({"success": True, "removed": True})


@app.route("/api/user/saved/clear", methods=["POST", "DELETE"])
def user_saved_clear():
    user = get_current_user_from_session()
    if not user:
        return jsonify({"error": "Authentication required.", "auth_required": True}), 401
    list_type = request.args.get("list_type")
    clear_user_saved_items(user["id"], list_type=list_type)
    return jsonify({"success": True})


@app.route("/api/user/ratings", methods=["GET", "POST"])
def user_ratings():
    user = get_current_user_from_session()
    if not user:
        return jsonify({"error": "Authentication required.", "auth_required": True}), 401

    if request.method == "GET":
        ratings = get_user_ratings(user["id"])
        return jsonify({"ratings": ratings})
    else:
        data = request.get_json(force=True, silent=True) or {}
        title = data.get("title")
        item_type = data.get("type", "movie")
        tier = data.get("tier", "good")
        if not title:
            return jsonify({"error": "Title is required"}), 400
        set_user_rating(user["id"], title, item_type, tier)
        return jsonify({"success": True})


# -------------------------------------------------------------
# QUICK EXPLORE ENDPOINTS (EXPLAINABLE RANKINGS)
# -------------------------------------------------------------

@app.route("/api/quick-explore", methods=["GET"])
def quick_explore():
    explore_type = request.args.get("type", "top_movies").lower().strip()
    
    if explore_type == "top_movies":
        data = get_top_10_movies(recommender.df_movies)
    elif explore_type == "top_songs":
        data = get_top_10_songs(recommender.df_songs)
    elif explore_type == "trending_artists":
        data = get_trending_artists(recommender.df_movies, recommender.df_songs)
    elif explore_type == "latest_releases":
        data = get_latest_releases(recommender.df_movies, recommender.df_songs)
    else:
        return jsonify({"error": f"Unknown explore type: {explore_type}"}), 400

    return jsonify(data)


# -------------------------------------------------------------
# SEARCH & RECOMMENDATION ENDPOINTS
# -------------------------------------------------------------

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

