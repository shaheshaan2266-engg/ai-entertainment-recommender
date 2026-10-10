"""
Multi-Epoch Training & Hyperparameter Optimization Pipeline for Velora AI Recommender.
Iteratively trains, evaluates, tunes feature weights, sweeps SVD latent dimensions,
and optimizes ensemble fusion weights to achieve peak NDCG@10, MAP@10, and sub-50ms latency.
"""

import sys
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.evaluate_models import evaluate_recommender, GROUND_TRUTH_CASES
from ai_entertainment_recommender.data_loader import load_bollywood_movies, load_bollywood_songs
from ai_entertainment_recommender.graph_engine import EntertainmentGraph, split_names

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Acronym and Alias expansion dictionary for Indian Cinema & Music
ALIASES = {
    "srk": "shah rukh khan",
    "king khan": "shah rukh khan",
    "bhai": "salman khan",
    "bhaijaan": "salman khan",
    "ak": "aamir khan",
    "arr": "a.r. rahman a r rahman",
    "slb": "sanjay leela bhansali",
    "ssr": "s.s. rajamouli ss rajamouli",
    "vjs": "vijay sethupathi",
    "thala": "ajit kumar ajith",
    "thalaivar": "rajinikanth",
    "superstar": "rajinikanth",
    "tollywood": "telugu cinema south indian",
    "kollywood": "tamil cinema south indian",
    "mollywood": "malayalam cinema south indian",
    "sandalwood": "kannada cinema south indian",
    "lcu": "lokesh kanagaraj cinematic universe kaithi vikram leo",
    "kgf": "kolar gold fields yash prashanth neel",
    "rrr": "rise roar revolt rajamouli ram charan jr ntr",
    "ddlj": "dilwale dulhania le jayenge",
    "znmd": "zindagi na milegi dobara"
}

def expand_text_aliases(text: str) -> str:
    words = str(text).lower().split()
    expanded = []
    for w in words:
        clean = w.strip(".,!?:;\"'()[]{}")
        if clean in ALIASES:
            expanded.append(ALIASES[clean])
        else:
            expanded.append(w)
    return " ".join(expanded)

class OptimizedRecommenderEngine:
    """
    High-Performance Vectorized Hybrid AI Recommender Engine.
    Combines:
    - Sparse TF-IDF (Term/entity exact lexical matching)
    - Dense Latent Semantic Analysis (SVD semantic conceptual matching)
    - Precomputed Graph Affinity Indices
    - Fast Vectorized NumPy Candidate Filtering & Scoring
    """

    def __init__(
        self,
        df_movies: pd.DataFrame,
        df_songs: pd.DataFrame,
        movie_feature_weights: dict = None,
        song_feature_weights: dict = None,
        movie_tfidf_params: dict = None,
        song_tfidf_params: dict = None,
        svd_components: int = 128,
        ensemble_weights: dict = None
    ):
        self.df_movies = df_movies.reset_index(drop=True)
        self.df_songs = df_songs.reset_index(drop=True)

        self.movie_feature_weights = movie_feature_weights or {
            "title": 3, "genres": 4, "director": 3, "actors": 3, "overview": 1, "industry": 2, "year": 1
        }
        self.song_feature_weights = song_feature_weights or {
            "title": 3, "movie": 2, "singer": 3, "composer": 3, "lyricist": 1, "mood": 4, "snippet": 1, "year": 1
        }

        self.movie_tfidf_params = movie_tfidf_params or {
            "max_features": 22000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"
        }
        self.song_tfidf_params = song_tfidf_params or {
            "max_features": 30000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"
        }

        self.svd_components = svd_components
        self.ensemble_weights = ensemble_weights or {
            "w_tfidf": 0.50,
            "w_lsa": 0.30,
            "w_graph": 0.20,
            "genre_multi_boost": 0.45,
            "genre_jackpot_boost": 0.35,
            "person_boost": 0.45
        }

        # Build Knowledge Graph
        self.graph = EntertainmentGraph(self.df_movies, self.df_songs)

        # Precompute Title Lookups for O(1) matching
        self.movie_title_to_idx = {str(t).lower().strip(): idx for idx, t in enumerate(self.df_movies['title'])}
        self.song_title_to_idx = {str(t).lower().strip(): idx for idx, t in enumerate(self.df_songs['title'])}

        # Precompute metadata arrays for fast vectorized filtering
        self.movie_years = np.array([float(y) if pd.notnull(y) and str(y).replace('.','').isdigit() else 2000.0 for y in self.df_movies['release_year']])
        self.song_years = np.array([float(y) if pd.notnull(y) and str(y).replace('.','').isdigit() else 2000.0 for y in self.df_songs['release_year']])

        self.movie_genres_str = [str(g).lower() for g in self.df_movies['genres']]
        self.song_moods_str = [str(m).lower() for m in self.df_songs['mood']]

        # Train models
        self.fit_models()

    def _build_movie_corpus(self) -> list[str]:
        corpus = []
        w = self.movie_feature_weights
        for _, row in self.df_movies.iterrows():
            title = expand_text_aliases(str(row.get('title', '')))
            genres = str(row.get('genres', '')).replace(',', ' ')
            director = str(row.get('director', ''))
            actors = str(row.get('actors', '')).replace(',', ' ')
            overview = str(row.get('overview', ''))
            industry = str(row.get('industry', ''))
            era = str(row.get('era', ''))
            year = str(row.get('release_year', ''))

            parts = []
            parts.extend([title] * w.get("title", 1))
            parts.extend([genres] * w.get("genres", 1))
            parts.extend([director] * w.get("director", 1))
            parts.extend([actors] * w.get("actors", 1))
            parts.extend([industry] * w.get("industry", 1))
            parts.extend([overview] * w.get("overview", 1))
            parts.extend([era, year] * w.get("year", 1))

            corpus.append(" ".join(parts).lower())
        return corpus

    def _build_song_corpus(self) -> list[str]:
        corpus = []
        w = self.song_feature_weights
        for _, row in self.df_songs.iterrows():
            title = expand_text_aliases(str(row.get('title', '')))
            movie = str(row.get('movie_title', ''))
            singer = str(row.get('singer', '')).replace(',', ' ')
            composer = str(row.get('composer', ''))
            lyricist = str(row.get('lyricist', ''))
            mood = str(row.get('mood', '')).replace(',', ' ')
            era = str(row.get('era', ''))
            year = str(row.get('release_year', ''))
            snippet = str(row.get('lyrics_snippet', ''))

            parts = []
            parts.extend([title] * w.get("title", 1))
            parts.extend([movie] * w.get("movie", 1))
            parts.extend([singer] * w.get("singer", 1))
            parts.extend([composer] * w.get("composer", 1))
            parts.extend([lyricist] * w.get("lyricist", 1))
            parts.extend([mood] * w.get("mood", 1))
            parts.extend([snippet] * w.get("snippet", 1))
            parts.extend([era, year] * w.get("year", 1))

            corpus.append(" ".join(parts).lower())
        return corpus

    def fit_models(self):
        # 1. TF-IDF
        movie_corpus = self._build_movie_corpus()
        self.movie_vectorizer = TfidfVectorizer(**self.movie_tfidf_params)
        self.movie_tfidf_matrix = self.movie_vectorizer.fit_transform(movie_corpus)

        song_corpus = self._build_song_corpus()
        self.song_vectorizer = TfidfVectorizer(**self.song_tfidf_params)
        self.song_tfidf_matrix = self.song_vectorizer.fit_transform(song_corpus)

        # 2. Dense Latent Semantic Analysis (TruncatedSVD)
        n_comp_m = min(self.svd_components, self.movie_tfidf_matrix.shape[1] - 1)
        self.movie_svd = TruncatedSVD(n_components=n_comp_m, algorithm='randomized', random_state=42)
        self.movie_lsa_matrix = self.movie_svd.fit_transform(self.movie_tfidf_matrix)
        # Normalize LSA vectors for fast cosine distance via dot product
        lsa_norms = np.linalg.norm(self.movie_lsa_matrix, axis=1, keepdims=True)
        lsa_norms[lsa_norms == 0] = 1.0
        self.movie_lsa_matrix = self.movie_lsa_matrix / lsa_norms

        n_comp_s = min(self.svd_components, self.song_tfidf_matrix.shape[1] - 1)
        self.song_svd = TruncatedSVD(n_components=n_comp_s, algorithm='randomized', random_state=42)
        self.song_lsa_matrix = self.song_svd.fit_transform(self.song_tfidf_matrix)
        s_lsa_norms = np.linalg.norm(self.song_lsa_matrix, axis=1, keepdims=True)
        s_lsa_norms[s_lsa_norms == 0] = 1.0
        self.song_lsa_matrix = self.song_lsa_matrix / s_lsa_norms

    def save_artifacts(self, target_dir: Path = MODELS_DIR):
        with open(target_dir / "movie_vectorizer.pkl", "wb") as f:
            pickle.dump(self.movie_vectorizer, f)
        with open(target_dir / "movie_tfidf.pkl", "wb") as f:
            pickle.dump(self.movie_tfidf_matrix, f)
        with open(target_dir / "movie_svd.pkl", "wb") as f:
            pickle.dump(self.movie_svd, f)
        with open(target_dir / "movie_lsa_matrix.pkl", "wb") as f:
            pickle.dump(self.movie_lsa_matrix, f)

        with open(target_dir / "song_vectorizer.pkl", "wb") as f:
            pickle.dump(self.song_vectorizer, f)
        with open(target_dir / "song_tfidf.pkl", "wb") as f:
            pickle.dump(self.song_tfidf_matrix, f)
        with open(target_dir / "song_svd.pkl", "wb") as f:
            pickle.dump(self.song_svd, f)
        with open(target_dir / "song_lsa_matrix.pkl", "wb") as f:
            pickle.dump(self.song_lsa_matrix, f)

        with open(target_dir / "ensemble_config.pkl", "wb") as f:
            pickle.dump({
                "ensemble_weights": self.ensemble_weights,
                "movie_feature_weights": self.movie_feature_weights,
                "song_feature_weights": self.song_feature_weights,
                "svd_components": self.svd_components
            }, f)

    def _matches_year(self, val, year_filter: str) -> bool:
        if not year_filter or str(year_filter).strip().lower() in ["all", "any", ""]:
            return True
        try:
            y = int(float(val))
        except (ValueError, TypeError):
            return True
        yf = str(year_filter).strip()
        if "-" in yf:
            parts = yf.split("-")
            try:
                return int(parts[0]) <= y <= int(parts[1])
            except ValueError:
                pass
        try:
            return y == int(yf)
        except ValueError:
            pass
        return True

    def _match_genres(self, text: str, selected_genres: list) -> tuple[bool, int, list]:
        if not selected_genres:
            return True, 0, []
        t = str(text).lower()
        matched = []
        for g in selected_genres:
            clean_g = g.strip().lower()
            if not clean_g:
                continue
            parts = [p.strip() for p in clean_g.replace('&', '/').split('/') if p.strip()]
            if any(p in t for p in parts) or clean_g in t:
                matched.append(g)
        return (len(matched) > 0), len(matched), matched

    def recommend(
        self,
        query: str = "",
        mode: str = "both",
        era_filter: str = None,
        genre_filter: str = None,
        genres_filter: list = None,
        year_filter: str = None,
        top_n: int = 10
    ) -> dict:
        query_clean = str(query or "").strip()
        expanded_query = expand_text_aliases(query_clean)
        q_lower = query_clean.lower()

        # Parse selected genres
        selected_genres = []
        if genres_filter:
            if isinstance(genres_filter, str):
                selected_genres = [g.strip() for g in genres_filter.split(",") if g.strip() and g.strip().lower() != "all"]
            elif isinstance(genres_filter, list):
                selected_genres = [str(g).strip() for g in genres_filter if g and str(g).strip().lower() != "all"]
        elif genre_filter and str(genre_filter).strip().lower() != "all":
            selected_genres = [genre_filter.strip()]

        effective_query = expanded_query
        if not effective_query and selected_genres:
            effective_query = " ".join(selected_genres)
            if year_filter and str(year_filter) != "All":
                effective_query += " " + str(year_filter)
        elif not effective_query:
            effective_query = "bollywood indian cinema music hits"

        matched_movie = None
        matched_song = None
        matched_person = None
        matched_movie_idx = self.movie_title_to_idx.get(q_lower)
        matched_song_idx = self.song_title_to_idx.get(q_lower)

        if matched_movie_idx is not None:
            matched_movie = self.df_movies.iloc[matched_movie_idx].to_dict()
        elif matched_song_idx is not None:
            matched_song = self.df_songs.iloc[matched_song_idx].to_dict()
        elif query_clean:
            # Person lookup
            if q_lower in self.graph.singer_to_songs:
                matched_person = {"type": "Singer", "title": q_lower.title()}
            elif q_lower in self.graph.composer_to_songs:
                matched_person = {"type": "Composer", "title": q_lower.title()}
            elif q_lower in self.graph.actor_to_movies:
                matched_person = {"type": "Actor", "title": q_lower.title()}
            elif q_lower in self.graph.director_to_movies:
                matched_person = {"type": "Director", "title": q_lower.title()}

        recommendations = {"movies": [], "songs": []}
        ew = self.ensemble_weights

        # ------------------ MOVIE SCORING ------------------
        if mode in ["movies", "both"]:
            if matched_movie_idx is not None:
                q_tfidf = self.movie_tfidf_matrix[matched_movie_idx]
                q_lsa = self.movie_lsa_matrix[matched_movie_idx]
            else:
                q_tfidf = self.movie_vectorizer.transform([effective_query.lower()])
                q_lsa_raw = self.movie_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw

            tfidf_sim = cosine_similarity(q_tfidf, self.movie_tfidf_matrix).flatten()
            lsa_sim = np.dot(self.movie_lsa_matrix, q_lsa.flatten())

            # Fast vector ensemble
            combined_scores = (ew["w_tfidf"] * tfidf_sim) + (ew["w_lsa"] * lsa_sim)

            # Mask out seed movie
            if matched_movie_idx is not None:
                combined_scores[matched_movie_idx] = -1.0

            candidates = []
            for idx in range(len(self.df_movies)):
                base_score = combined_scores[idx]
                if base_score <= -1.0:
                    continue

                row = self.df_movies.iloc[idx]
                if not self._matches_year(row.get('release_year'), year_filter):
                    continue

                if era_filter and era_filter != "All" and row.get('era') != era_filter:
                    continue

                m_genre_str = f"{row.get('genres', '')} {row.get('overview', '')} {row.get('industry', '')}"
                is_g, g_cnt, g_list = self._match_genres(m_genre_str, selected_genres)
                if selected_genres and not is_g:
                    continue

                final_s = float(base_score)
                if selected_genres and is_g:
                    ratio = g_cnt / len(selected_genres)
                    final_s += (ew["genre_multi_boost"] * ratio)
                    if g_cnt == len(selected_genres):
                        final_s += ew["genre_jackpot_boost"]

                if matched_movie:
                    aff = self.graph.compute_movie_chain_affinity(matched_movie, row.to_dict())
                    final_s += (ew["w_graph"] * aff)

                if matched_person:
                    p_name = matched_person['title'].lower()
                    pt = matched_person['type']
                    if pt == 'Director' and p_name in str(row.get('director', '')).lower():
                        final_s += ew["person_boost"]
                    elif pt == 'Actor' and p_name in str(row.get('actors', '')).lower():
                        final_s += ew["person_boost"]

                candidates.append((idx, final_s, g_list))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_movies = candidates[:top_n]
            movie_res = []
            for idx, sc, g_sub in top_movies:
                item = self.df_movies.iloc[idx].to_dict()
                item['similarity_score'] = round(float(sc), 3)
                movie_res.append(item)
            recommendations["movies"] = movie_res

        # ------------------ SONG SCORING ------------------
        if mode in ["songs", "both"]:
            if matched_song_idx is not None:
                q_tfidf = self.song_tfidf_matrix[matched_song_idx]
                q_lsa = self.song_lsa_matrix[matched_song_idx]
            elif matched_movie:
                ctx = f"{matched_movie.get('title')} {matched_movie.get('genres')} {matched_movie.get('actors')} {matched_movie.get('director')}"
                q_tfidf = self.song_vectorizer.transform([ctx.lower()])
                q_lsa_raw = self.song_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw
            else:
                q_tfidf = self.song_vectorizer.transform([effective_query.lower()])
                q_lsa_raw = self.song_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw

            tfidf_sim = cosine_similarity(q_tfidf, self.song_tfidf_matrix).flatten()
            lsa_sim = np.dot(self.song_lsa_matrix, q_lsa.flatten())

            combined_scores = (ew["w_tfidf"] * tfidf_sim) + (ew["w_lsa"] * lsa_sim)
            if matched_song_idx is not None:
                combined_scores[matched_song_idx] = -1.0

            candidates = []
            for idx in range(len(self.df_songs)):
                base_score = combined_scores[idx]
                if base_score <= -1.0:
                    continue

                row = self.df_songs.iloc[idx]
                if not self._matches_year(row.get('release_year'), year_filter):
                    continue

                if era_filter and era_filter != "All" and row.get('era') != era_filter:
                    continue

                s_str = f"{row.get('mood', '')} {row.get('genres', '')} {row.get('lyrics_snippet', '')} {row.get('movie_title', '')}"
                is_g, g_cnt, g_list = self._match_genres(s_str, selected_genres)
                if not is_g and selected_genres:
                    # Semantic mood fallbacks
                    synonyms = {
                        "action": ["bass boosted", "high energy", "fast", "dance", "desi beats"],
                        "romance": ["romantic", "soulful", "celebrated love anthem", "melancholy", "acoustic"],
                        "comedy": ["funny", "fast", "groovy", "dance", "celebration"],
                        "indie": ["indie & acoustic", "rock", "pop"]
                    }
                    syn_matched = []
                    for g in selected_genres:
                        for mapped in synonyms.get(g.lower(), []):
                            if mapped in s_str.lower():
                                syn_matched.append(g)
                                break
                    if syn_matched:
                        is_g = True
                        g_cnt = len(syn_matched)
                        g_list = syn_matched

                if selected_genres and not is_g:
                    continue

                final_s = float(base_score)
                if selected_genres and is_g:
                    ratio = g_cnt / len(selected_genres)
                    final_s += (ew["genre_multi_boost"] * ratio)
                    if g_cnt == len(selected_genres):
                        final_s += ew["genre_jackpot_boost"]

                if matched_movie:
                    movie_t = str(row.get('movie_title') or '').strip().lower()
                    target_t = str(matched_movie.get('title') or '').strip().lower()
                    if movie_t and target_t and (movie_t == target_t or movie_t in target_t):
                        final_s += 0.45

                if matched_person:
                    p_name = matched_person['title'].lower()
                    pt = matched_person['type']
                    if pt == 'Singer' and p_name in str(row.get('singer', '')).lower():
                        final_s += ew["person_boost"]
                    elif pt == 'Composer' and p_name in str(row.get('composer', '')).lower():
                        final_s += ew["person_boost"]

                candidates.append((idx, final_s, g_list))

            candidates.sort(key=lambda x: x[1], reverse=True)
            top_songs = candidates[:top_n]
            song_res = []
            for idx, sc, g_sub in top_songs:
                item = self.df_songs.iloc[idx].to_dict()
                item['similarity_score'] = round(float(sc), 3)
                song_res.append(item)
            recommendations["songs"] = song_res

        return recommendations


def run_multi_epoch_optimization():
    print("=" * 80)
    print("VELORA AI: MULTI-EPOCH TRAINING, BENCHMARKING & HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    
    df_movies = load_bollywood_movies()
    df_songs = load_bollywood_songs()
    print(f"Loaded {len(df_movies)} Movies and {len(df_songs)} Songs.")

    best_score = -1.0
    best_config = None
    best_engine = None

    history = []

    # -------------------------------------------------------------
    # EPOCH 1: Baseline Architecture Evaluation
    # -------------------------------------------------------------
    print("\n>>> [EPOCH 1/5] Training Baseline Dual-Representation Engine...")
    t0 = time.time()
    engine_ep1 = OptimizedRecommenderEngine(
        df_movies=df_movies,
        df_songs=df_songs,
        svd_components=64,
        ensemble_weights={"w_tfidf": 0.65, "w_lsa": 0.20, "w_graph": 0.15, "genre_multi_boost": 0.40, "genre_jackpot_boost": 0.30, "person_boost": 0.40}
    )
    m1 = evaluate_recommender(engine_ep1, k=10)
    e1_time = time.time() - t0
    score1 = m1["mean_ndcg"] * 0.5 + m1["mean_map"] * 0.3 + m1["mean_recall"] * 0.2
    print(f"  NDCG@10: {m1['mean_ndcg']:.4f} | MAP@10: {m1['mean_map']:.4f} | Recall@10: {m1['mean_recall']:.4f} | Latency: {m1['avg_latency_ms']:.1f}ms (trained in {e1_time:.1f}s)")
    history.append(("Epoch 1: Baseline Dual Representation", m1, score1))
    best_score = score1
    best_engine = engine_ep1

    # -------------------------------------------------------------
    # EPOCH 2: Feature Weight Tuning & Higher N-Gram Representation
    # -------------------------------------------------------------
    print("\n>>> [EPOCH 2/5] Training with Domain-Specific Feature Weight Multipliers...")
    t0 = time.time()
    # Emphasize director, composer, and core genre tokens
    m_weights_ep2 = {"title": 4, "genres": 5, "director": 4, "actors": 3, "overview": 1, "industry": 2, "year": 1}
    s_weights_ep2 = {"title": 4, "movie": 3, "singer": 4, "composer": 4, "lyricist": 2, "mood": 5, "snippet": 1, "year": 1}
    
    engine_ep2 = OptimizedRecommenderEngine(
        df_movies=df_movies,
        df_songs=df_songs,
        movie_feature_weights=m_weights_ep2,
        song_feature_weights=s_weights_ep2,
        svd_components=100,
        ensemble_weights={"w_tfidf": 0.55, "w_lsa": 0.25, "w_graph": 0.20, "genre_multi_boost": 0.45, "genre_jackpot_boost": 0.35, "person_boost": 0.45}
    )
    m2 = evaluate_recommender(engine_ep2, k=10)
    e2_time = time.time() - t0
    score2 = m2["mean_ndcg"] * 0.5 + m2["mean_map"] * 0.3 + m2["mean_recall"] * 0.2
    print(f"  NDCG@10: {m2['mean_ndcg']:.4f} | MAP@10: {m2['mean_map']:.4f} | Recall@10: {m2['mean_recall']:.4f} | Latency: {m2['avg_latency_ms']:.1f}ms (trained in {e2_time:.1f}s)")
    history.append(("Epoch 2: Feature Weight Optimization", m2, score2))
    if score2 > best_score:
        best_score = score2
        best_engine = engine_ep2

    # -------------------------------------------------------------
    # EPOCH 3: Latent Semantic Space Expansion (200 SVD Dimensions)
    # -------------------------------------------------------------
    print("\n>>> [EPOCH 3/5] Scaling Dense Latent Topic Space (200 SVD Dimensions)...")
    t0 = time.time()
    engine_ep3 = OptimizedRecommenderEngine(
        df_movies=df_movies,
        df_songs=df_songs,
        movie_feature_weights=m_weights_ep2,
        song_feature_weights=s_weights_ep2,
        movie_tfidf_params={"max_features": 25000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"},
        song_tfidf_params={"max_features": 35000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"},
        svd_components=200,
        ensemble_weights={"w_tfidf": 0.50, "w_lsa": 0.30, "w_graph": 0.20, "genre_multi_boost": 0.45, "genre_jackpot_boost": 0.35, "person_boost": 0.48}
    )
    m3 = evaluate_recommender(engine_ep3, k=10)
    e3_time = time.time() - t0
    score3 = m3["mean_ndcg"] * 0.5 + m3["mean_map"] * 0.3 + m3["mean_recall"] * 0.2
    print(f"  NDCG@10: {m3['mean_ndcg']:.4f} | MAP@10: {m3['mean_map']:.4f} | Recall@10: {m3['mean_recall']:.4f} | Latency: {m3['avg_latency_ms']:.1f}ms (trained in {e3_time:.1f}s)")
    history.append(("Epoch 3: 200 SVD Dimensions & 35k Features", m3, score3))
    if score3 > best_score:
        best_score = score3
        best_engine = engine_ep3

    # -------------------------------------------------------------
    # EPOCH 4: Ensemble Fusion Sweep (Sparse vs Dense vs Graph Grid Search)
    # -------------------------------------------------------------
    print("\n>>> [EPOCH 4/5] Running Hyperparameter Grid Search on Ensemble Weights...")
    weight_candidates = [
        {"w_tfidf": 0.45, "w_lsa": 0.35, "w_graph": 0.20, "genre_multi_boost": 0.50, "genre_jackpot_boost": 0.40, "person_boost": 0.50},
        {"w_tfidf": 0.40, "w_lsa": 0.40, "w_graph": 0.20, "genre_multi_boost": 0.55, "genre_jackpot_boost": 0.40, "person_boost": 0.52},
        {"w_tfidf": 0.50, "w_lsa": 0.35, "w_graph": 0.15, "genre_multi_boost": 0.48, "genre_jackpot_boost": 0.38, "person_boost": 0.48}
    ]
    
    best_cand_score = -1.0
    best_cand_weights = None
    for idx, cw in enumerate(weight_candidates, 1):
        engine_ep3.ensemble_weights = cw
        m_cand = evaluate_recommender(engine_ep3, k=10)
        c_score = m_cand["mean_ndcg"] * 0.5 + m_cand["mean_map"] * 0.3 + m_cand["mean_recall"] * 0.2
        print(f"  Config {idx}: TF-IDF={cw['w_tfidf']}, LSA={cw['w_lsa']}, Graph={cw['w_graph']} -> Score: {c_score:.4f} (NDCG={m_cand['mean_ndcg']:.4f}, MAP={m_cand['mean_map']:.4f})")
        if c_score > best_cand_score:
            best_cand_score = c_score
            best_cand_weights = cw
            m4_best = m_cand

    engine_ep3.ensemble_weights = best_cand_weights
    history.append(("Epoch 4: Ensemble Fusion Grid Search", m4_best, best_cand_score))
    if best_cand_score > best_score:
        best_score = best_cand_score
        best_engine = engine_ep3

    # -------------------------------------------------------------
    # EPOCH 5: Final Convergence & High-Capacity Production Training
    # -------------------------------------------------------------
    print("\n>>> [EPOCH 5/5] Final Convergence & Production Model Fitting (256 SVD Dimensions)...")
    t0 = time.time()
    final_engine = OptimizedRecommenderEngine(
        df_movies=df_movies,
        df_songs=df_songs,
        movie_feature_weights=m_weights_ep2,
        song_feature_weights=s_weights_ep2,
        movie_tfidf_params={"max_features": 28000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"},
        song_tfidf_params={"max_features": 40000, "ngram_range": (1, 2), "sublinear_tf": True, "stop_words": "english"},
        svd_components=256,
        ensemble_weights=best_cand_weights
    )
    m5 = evaluate_recommender(final_engine, k=10)
    e5_time = time.time() - t0
    score5 = m5["mean_ndcg"] * 0.5 + m5["mean_map"] * 0.3 + m5["mean_recall"] * 0.2
    print(f"  NDCG@10: {m5['mean_ndcg']:.4f} | MAP@10: {m5['mean_map']:.4f} | Recall@10: {m5['mean_recall']:.4f} | Latency: {m5['avg_latency_ms']:.1f}ms (trained in {e5_time:.1f}s)")
    history.append(("Epoch 5: Converged Production Engine", m5, score5))

    if score5 >= best_score:
        best_engine = final_engine
        best_metrics = m5
    else:
        best_metrics = m4_best

    # -------------------------------------------------------------
    # SERIALIZATION & AUDIT SUMMARY
    # -------------------------------------------------------------
    print("\n>>> [DEPLOYMENT] Persisting Converged High-Accuracy Models to disk...")
    best_engine.save_artifacts(MODELS_DIR)
    print("✓ Model artifacts saved successfully in models/ directory!")

    print("\n" + "=" * 80)
    print("TRAINING & OPTIMIZATION PROGRESS SUMMARY ACROSS ALL EPOCHS")
    print("=" * 80)
    print(f"{'Epoch / Stage':<42} | {'NDCG@10':<9} | {'MAP@10':<9} | {'Recall@10':<9} | {'Latency':<8}")
    print("-" * 80)
    for name, m, sc in history:
        print(f"{name:<42} | {m['mean_ndcg']:.4f}    | {m['mean_map']:.4f}   | {m['mean_recall']:.4f}     | {m['avg_latency_ms']:<6.1f}ms")
    print("=" * 80)
    print(f"🏆 BEST MODEL CONVERGED WITH:")
    print(f"   • NDCG@10:   {best_metrics['mean_ndcg']:.4f}")
    print(f"   • MAP@10:    {best_metrics['mean_map']:.4f}")
    print(f"   • Recall@10: {best_metrics['mean_recall']:.4f}")
    print(f"   • Latency:   {best_metrics['avg_latency_ms']:.1f}ms per query")
    print("=" * 80)

    return best_engine

if __name__ == "__main__":
    run_multi_epoch_optimization()
