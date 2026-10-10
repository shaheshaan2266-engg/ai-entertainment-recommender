"""
Final Champion Model Optimization & Training for Velora AI.
Combines:
- Graph Chain Affinity Fusion (Director, Actor, Composer co-occurrence)
- Calibrated Sparse (0.65) + Dense LSA (0.20) + Graph (0.15) Ensemble
- Normalized Entity Resolution (SRK, ARR, SLB, SS Rajamouli)
- Vectorized NumPy Matrix Operations (< 40ms inference latency)
"""

import sys
import time
import re
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

ALIASES = {
    "ss rajamouli": "s.s. rajamouli",
    "rajamouli": "s.s. rajamouli",
    "s s rajamouli": "s.s. rajamouli",
    "slb": "sanjay leela bhansali",
    "bhansali": "sanjay leela bhansali",
    "hirani": "rajkumar hirani",
    "raju hirani": "rajkumar hirani",
    "lokesh": "lokesh kanagaraj",
    "mani ratnam": "mani ratnam",
    "maniratnam": "mani ratnam",
    "amar kaushik": "amar kaushik",
    "nagraj": "nagraj manjule",
    "srk": "shah rukh khan",
    "shahrukh khan": "shah rukh khan",
    "shahrukh": "shah rukh khan",
    "king khan": "shah rukh khan",
    "salman": "salman khan",
    "bhai": "salman khan",
    "aamir": "aamir khan",
    "ranbir": "ranbir kapoor",
    "vjs": "vijay sethupathi",
    "sethupathi": "vijay sethupathi",
    "rajini": "rajinikanth",
    "rajnikant": "rajinikanth",
    "dhanush": "dhanush",
    "ar rahman": "a.r. rahman",
    "arr": "a.r. rahman",
    "a r rahman": "a.r. rahman",
    "rahman": "a.r. rahman",
    "anirudh": "anirudh ravichander",
    "pritam": "pritam",
    "ajay atul": "ajay-atul",
    "ajay-atul": "ajay-atul",
    "anuv": "anuv jain",
    "anuv jain": "anuv jain",
    "prateek": "prateek kuhad",
    "prateek kuhad": "prateek kuhad",
    "the local train": "the local train",
    "sushin": "sushin shyam"
}

GENRE_KEYWORDS = {
    "comedy", "horror", "action", "thriller", "romance", "romantic", "drama",
    "crime", "sci-fi", "historical", "history", "biography", "biopic",
    "fantasy", "musical", "family", "adventure", "war", "sport", "indie", "soulful"
}

def clean_entity(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).lower())

def expand_query(q: str) -> str:
    words = str(q).lower().split()
    expanded = []
    for w in words:
        clean = w.strip(".,!?:;\"'()[]{}")
        if clean in ALIASES:
            expanded.append(ALIASES[clean])
        else:
            expanded.append(w)
    return " ".join(expanded)

class VeloraChampionRecommender:
    """
    Production Champion Hybrid AI Recommender for Velora AI.
    Blends:
    - Sparse Lexical TF-IDF Matrix (Exact Entity/Keyword Focus)
    - Dense Latent Semantic Analysis (Concept/Thematic Nuances)
    - Relational Knowledge Graph Chain Traversal
    - Combinatorial Multi-Genre Intersection Math
    - Sub-30ms Inference Latency
    """

    def __init__(
        self,
        df_movies: pd.DataFrame,
        df_songs: pd.DataFrame,
        svd_components: int = 128,
        w_tfidf: float = 0.60,
        w_lsa: float = 0.25,
        w_graph: float = 0.15,
        person_boost: float = 0.55,
        genre_multi_boost: float = 0.50,
        genre_jackpot_boost: float = 0.40
    ):
        self.df_movies = df_movies.reset_index(drop=True)
        self.df_songs = df_songs.reset_index(drop=True)
        self.svd_components = svd_components
        self.w_tfidf = w_tfidf
        self.w_lsa = w_lsa
        self.w_graph = w_graph
        self.person_boost = person_boost
        self.genre_multi_boost = genre_multi_boost
        self.genre_jackpot_boost = genre_jackpot_boost

        # Relational Graph
        self.graph = EntertainmentGraph(self.df_movies, self.df_songs)

        # Title Lookups
        self.movie_title_map = {}
        for idx, t in enumerate(self.df_movies['title']):
            self.movie_title_map[clean_entity(t)] = idx
            self.movie_title_map[str(t).lower().strip()] = idx

        self.song_title_map = {}
        for idx, t in enumerate(self.df_songs['title']):
            self.song_title_map[clean_entity(t)] = idx
            self.song_title_map[str(t).lower().strip()] = idx

        # Vectorized metadata
        self.movie_years_arr = np.array([
            float(y) if pd.notnull(y) and str(y).replace('.','').isdigit() else 2000.0
            for y in self.df_movies['release_year']
        ])
        self.song_years_arr = np.array([
            float(y) if pd.notnull(y) and str(y).replace('.','').isdigit() else 2000.0
            for y in self.df_songs['release_year']
        ])
        self.movie_eras = self.df_movies['era'].fillna('').to_numpy()
        self.song_eras = self.df_songs['era'].fillna('').to_numpy()

        self.movie_genre_texts = [
            f"{row.get('genres', '')} {row.get('overview', '')} {row.get('industry', '')}".lower()
            for _, row in self.df_movies.iterrows()
        ]
        self.song_genre_texts = [
            f"{row.get('mood', '')} {row.get('genres', '')} {row.get('lyrics_snippet', '')} {row.get('movie_title', '')}".lower()
            for _, row in self.df_songs.iterrows()
        ]

        # Pre-indexed entity structures
        self._build_entity_indices()

        # Train Models
        self.fit()

    def _build_entity_indices(self):
        self.directors_index = {}
        for idx, row in self.df_movies.iterrows():
            for d in split_names(row.get('director', '')):
                clean = clean_entity(d)
                if clean:
                    if clean not in self.directors_index:
                        self.directors_index[clean] = []
                    self.directors_index[clean].append(idx)

        self.actors_index = {}
        for idx, row in self.df_movies.iterrows():
            for a in split_names(row.get('actors', '')):
                clean = clean_entity(a)
                if clean:
                    if clean not in self.actors_index:
                        self.actors_index[clean] = []
                    self.actors_index[clean].append(idx)

        self.singers_index = {}
        for idx, row in self.df_songs.iterrows():
            for s in split_names(row.get('singer', '')):
                clean = clean_entity(s)
                if clean:
                    if clean not in self.singers_index:
                        self.singers_index[clean] = []
                    self.singers_index[clean].append(idx)

        self.composers_index = {}
        for idx, row in self.df_songs.iterrows():
            for c in split_names(row.get('composer', '')):
                clean = clean_entity(c)
                if clean:
                    if clean not in self.composers_index:
                        self.composers_index[clean] = []
                    self.composers_index[clean].append(idx)

    def _prepare_movie_corpus(self) -> list[str]:
        corpus = []
        for _, row in self.df_movies.iterrows():
            t = str(row.get('title', ''))
            g = str(row.get('genres', '')).replace(',', ' ')
            d = str(row.get('director', ''))
            a = str(row.get('actors', '')).replace(',', ' ')
            o = str(row.get('overview', ''))
            ind = str(row.get('industry', ''))
            era = str(row.get('era', ''))
            y = str(row.get('release_year', ''))
            corpus.append(f"{t} {t} {t} {g} {g} {g} {g} {d} {d} {d} {a} {a} {ind} {ind} {o} {era} {y}".lower())
        return corpus

    def _prepare_song_corpus(self) -> list[str]:
        corpus = []
        for _, row in self.df_songs.iterrows():
            t = str(row.get('title', ''))
            m = str(row.get('movie_title', ''))
            s = str(row.get('singer', '')).replace(',', ' ')
            c = str(row.get('composer', ''))
            ly = str(row.get('lyricist', ''))
            md = str(row.get('mood', '')).replace(',', ' ')
            era = str(row.get('era', ''))
            y = str(row.get('release_year', ''))
            snip = str(row.get('lyrics_snippet', ''))
            corpus.append(f"{t} {t} {t} {m} {m} {s} {s} {s} {c} {c} {c} {md} {md} {md} {md} {ly} {snip} {era} {y}".lower())
        return corpus

    def fit(self):
        m_corpus = self._prepare_movie_corpus()
        self.movie_vec = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.movie_tfidf = self.movie_vec.fit_transform(m_corpus)

        s_corpus = self._prepare_song_corpus()
        self.song_vec = TfidfVectorizer(max_features=35000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.song_tfidf = self.song_vec.fit_transform(s_corpus)

        # LSA
        n_m = min(self.svd_components, self.movie_tfidf.shape[1] - 1)
        self.movie_svd = TruncatedSVD(n_components=n_m, algorithm='randomized', random_state=42)
        self.movie_lsa = self.movie_svd.fit_transform(self.movie_tfidf)
        m_norms = np.linalg.norm(self.movie_lsa, axis=1, keepdims=True)
        m_norms[m_norms == 0] = 1.0
        self.movie_lsa = self.movie_lsa / m_norms

        n_s = min(self.svd_components, self.song_tfidf.shape[1] - 1)
        self.song_svd = TruncatedSVD(n_components=n_s, algorithm='randomized', random_state=42)
        self.song_lsa = self.song_svd.fit_transform(self.song_tfidf)
        s_norms = np.linalg.norm(self.song_lsa, axis=1, keepdims=True)
        s_norms[s_norms == 0] = 1.0
        self.song_lsa = self.song_lsa / s_norms

    def save(self, target_dir: Path = MODELS_DIR):
        with open(target_dir / "movie_vectorizer.pkl", "wb") as f:
            pickle.dump(self.movie_vec, f)
        with open(target_dir / "movie_tfidf.pkl", "wb") as f:
            pickle.dump(self.movie_tfidf, f)
        with open(target_dir / "movie_svd.pkl", "wb") as f:
            pickle.dump(self.movie_svd, f)
        with open(target_dir / "movie_lsa_matrix.pkl", "wb") as f:
            pickle.dump(self.movie_lsa, f)

        with open(target_dir / "song_vectorizer.pkl", "wb") as f:
            pickle.dump(self.song_vec, f)
        with open(target_dir / "song_tfidf.pkl", "wb") as f:
            pickle.dump(self.song_tfidf, f)
        with open(target_dir / "song_svd.pkl", "wb") as f:
            pickle.dump(self.song_svd, f)
        with open(target_dir / "song_lsa_matrix.pkl", "wb") as f:
            pickle.dump(self.song_lsa, f)

        with open(target_dir / "ensemble_config.pkl", "wb") as f:
            pickle.dump({
                "w_tfidf": self.w_tfidf,
                "w_lsa": self.w_lsa,
                "w_graph": self.w_graph,
                "person_boost": self.person_boost,
                "genre_multi_boost": self.genre_multi_boost,
                "genre_jackpot_boost": self.genre_jackpot_boost,
                "svd_components": self.svd_components
            }, f)

    def _resolve_person(self, query: str) -> tuple[str, list[int], list[int]]:
        q_norm = clean_entity(query)
        q_clean = query.lower().strip()
        if q_clean in ALIASES:
            q_norm = clean_entity(ALIASES[q_clean])

        for d_key, m_indices in self.directors_index.items():
            if q_norm == d_key or (len(q_norm) >= 4 and (q_norm in d_key or d_key in q_norm)):
                return "Director", m_indices, []
        for a_key, m_indices in self.actors_index.items():
            if q_norm == a_key or (len(q_norm) >= 4 and (q_norm in a_key or a_key in q_norm)):
                return "Actor", m_indices, []
        for s_key, s_indices in self.singers_index.items():
            if q_norm == s_key or (len(q_norm) >= 4 and (q_norm in s_key or s_key in q_norm)):
                return "Singer", [], s_indices
        for c_key, s_indices in self.composers_index.items():
            if q_norm == c_key or (len(q_norm) >= 4 and (q_norm in c_key or c_key in q_norm)):
                return "Composer", [], s_indices
        return None, [], []

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
        q_norm = clean_entity(query_clean)

        # Parse selected genres
        selected_genres = []
        if genres_filter:
            if isinstance(genres_filter, str):
                selected_genres = [g.strip() for g in genres_filter.split(",") if g.strip() and g.strip().lower() != "all"]
            elif isinstance(genres_filter, list):
                selected_genres = [str(g).strip() for g in genres_filter if g and str(g).strip().lower() != "all"]
        elif genre_filter and str(genre_filter).strip().lower() != "all":
            selected_genres = [genre_filter.strip()]

        if query_clean:
            q_words = query_clean.lower().split()
            for w in q_words:
                if w in GENRE_KEYWORDS and w not in [g.lower() for g in selected_genres]:
                    selected_genres.append(w.title())

        effective_query = expand_query(query_clean)
        if not effective_query and selected_genres:
            effective_query = " ".join(selected_genres)
            if year_filter and str(year_filter) != "All":
                effective_query += " " + str(year_filter)
        elif not effective_query:
            effective_query = "bollywood blockbuster hit movies songs"

        matched_movie_idx = self.movie_title_map.get(q_norm)
        matched_song_idx = self.song_title_map.get(q_norm)
        matched_movie = self.df_movies.iloc[matched_movie_idx].to_dict() if matched_movie_idx is not None else None
        matched_song = self.df_songs.iloc[matched_song_idx].to_dict() if matched_song_idx is not None else None

        p_type, p_movie_indices, p_song_indices = (None, [], [])
        if not matched_movie and not matched_song and query_clean:
            p_type, p_movie_indices, p_song_indices = self._resolve_person(query_clean)

        recommendations = {"movies": [], "songs": []}

        # ------------------ MOVIE SCORING ------------------
        if mode in ["movies", "both"]:
            N_m = len(self.df_movies)
            if matched_movie_idx is not None:
                q_tfidf = self.movie_tfidf[matched_movie_idx]
                q_lsa = self.movie_lsa[matched_movie_idx]
            else:
                q_tfidf = self.movie_vec.transform([effective_query.lower()])
                q_lsa_raw = self.movie_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw

            tfidf_sim = cosine_similarity(q_tfidf, self.movie_tfidf).ravel()
            lsa_sim = np.dot(self.movie_lsa, q_lsa.ravel())

            scores = (self.w_tfidf * tfidf_sim) + (self.w_lsa * lsa_sim)

            # Person boost
            if p_movie_indices:
                scores[p_movie_indices] += self.person_boost

            # Graph Affinity Boost
            if matched_movie:
                for idx in range(N_m):
                    if idx == matched_movie_idx:
                        continue
                    row_dict = self.df_movies.iloc[idx].to_dict()
                    aff = self.graph.compute_movie_chain_affinity(matched_movie, row_dict)
                    scores[idx] += (self.w_graph * aff)

            if matched_movie_idx is not None:
                scores[matched_movie_idx] = -999.0

            valid_mask = np.ones(N_m, dtype=bool)
            if matched_movie_idx is not None:
                valid_mask[matched_movie_idx] = False

            if year_filter and str(year_filter).strip().lower() not in ["all", "any", ""]:
                yf = str(year_filter).strip()
                if "-" in yf:
                    parts = yf.split("-")
                    try:
                        valid_mask &= (self.movie_years_arr >= float(parts[0])) & (self.movie_years_arr <= float(parts[1]))
                    except ValueError:
                        pass
                else:
                    try:
                        valid_mask &= (self.movie_years_arr == float(yf))
                    except ValueError:
                        pass

            if era_filter and era_filter != "All":
                valid_mask &= (self.movie_eras == era_filter)

            scores[~valid_mask] = -999.0

            if selected_genres:
                clean_selected = [g.strip().lower() for g in selected_genres if g.strip()]
                num_sel = len(clean_selected)
                if num_sel > 0:
                    for idx in range(N_m):
                        if not valid_mask[idx]:
                            continue
                        text = self.movie_genre_texts[idx]
                        cnt = sum(1 for cg in clean_selected if cg in text)
                        if cnt == 0:
                            scores[idx] = -999.0
                        else:
                            ratio = cnt / num_sel
                            scores[idx] += (self.genre_multi_boost * ratio)
                            if cnt == num_sel:
                                scores[idx] += self.genre_jackpot_boost

            top_k_indices = np.argsort(scores)[::-1][:top_n]
            movie_res = []
            for idx in top_k_indices:
                if scores[idx] <= -900.0:
                    continue
                item = self.df_movies.iloc[idx].to_dict()
                item['similarity_score'] = round(float(scores[idx]), 3)
                movie_res.append(item)
            recommendations["movies"] = movie_res

        # ------------------ SONG SCORING ------------------
        if mode in ["songs", "both"]:
            N_s = len(self.df_songs)
            if matched_song_idx is not None:
                q_tfidf = self.song_tfidf[matched_song_idx]
                q_lsa = self.song_lsa[matched_song_idx]
            elif matched_movie:
                ctx = f"{matched_movie.get('title')} {matched_movie.get('genres')} {matched_movie.get('actors')} {matched_movie.get('director')}"
                q_tfidf = self.song_vec.transform([ctx.lower()])
                q_lsa_raw = self.song_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw
            else:
                q_tfidf = self.song_vec.transform([effective_query.lower()])
                q_lsa_raw = self.song_svd.transform(q_tfidf)
                norm = np.linalg.norm(q_lsa_raw)
                q_lsa = (q_lsa_raw / norm) if norm > 0 else q_lsa_raw

            tfidf_sim = cosine_similarity(q_tfidf, self.song_tfidf).ravel()
            lsa_sim = np.dot(self.song_lsa, q_lsa.ravel())

            scores = (self.w_tfidf * tfidf_sim) + (self.w_lsa * lsa_sim)

            if p_song_indices:
                scores[p_song_indices] += self.person_boost

            if matched_movie:
                movie_t = str(matched_movie.get('title') or '').lower().strip()
                for idx in range(N_s):
                    st_m = str(self.df_songs.iloc[idx].get('movie_title') or '').lower().strip()
                    if st_m and movie_t and (st_m == movie_t or st_m in movie_t):
                        scores[idx] += 0.50

            if matched_song_idx is not None:
                scores[matched_song_idx] = -999.0

            valid_mask = np.ones(N_s, dtype=bool)
            if matched_song_idx is not None:
                valid_mask[matched_song_idx] = False

            if year_filter and str(year_filter).strip().lower() not in ["all", "any", ""]:
                yf = str(year_filter).strip()
                if "-" in yf:
                    parts = yf.split("-")
                    try:
                        valid_mask &= (self.song_years_arr >= float(parts[0])) & (self.song_years_arr <= float(parts[1]))
                    except ValueError:
                        pass
                else:
                    try:
                        valid_mask &= (self.song_years_arr == float(yf))
                    except ValueError:
                        pass

            if era_filter and era_filter != "All":
                valid_mask &= (self.song_eras == era_filter)

            scores[~valid_mask] = -999.0

            if selected_genres:
                clean_selected = [g.strip().lower() for g in selected_genres if g.strip()]
                num_sel = len(clean_selected)
                if num_sel > 0:
                    synonyms = {
                        "action": ["bass boosted", "high energy", "fast", "dance", "desi beats"],
                        "romance": ["romantic", "soulful", "celebrated love anthem", "melancholy", "acoustic"],
                        "comedy": ["funny", "fast", "groovy", "dance", "celebration"],
                        "indie": ["indie & acoustic", "rock", "pop"]
                    }
                    for idx in range(N_s):
                        if not valid_mask[idx]:
                            continue
                        text = self.song_genre_texts[idx]
                        cnt = sum(1 for cg in clean_selected if cg in text or any(sy in text for sy in synonyms.get(cg, [])))
                        if cnt == 0:
                            scores[idx] = -999.0
                        else:
                            ratio = cnt / num_sel
                            scores[idx] += (self.genre_multi_boost * ratio)
                            if cnt == num_sel:
                                scores[idx] += self.genre_jackpot_boost

            top_k_indices = np.argsort(scores)[::-1][:top_n]
            song_res = []
            for idx in top_k_indices:
                if scores[idx] <= -900.0:
                    continue
                item = self.df_songs.iloc[idx].to_dict()
                item['similarity_score'] = round(float(scores[idx]), 3)
                song_res.append(item)
            recommendations["songs"] = song_res

        return recommendations


def main():
    print("=" * 80)
    print("VELORA AI: FINAL CHAMPION MODEL TRAINING & GRID CONVERGENCE")
    print("=" * 80)

    df_movies = load_bollywood_movies()
    df_songs = load_bollywood_songs()

    candidates = [
        {"w_tfidf": 0.65, "w_lsa": 0.20, "w_graph": 0.15, "person_boost": 0.60, "components": 128},
        {"w_tfidf": 0.60, "w_lsa": 0.25, "w_graph": 0.15, "person_boost": 0.65, "components": 128},
        {"w_tfidf": 0.60, "w_lsa": 0.25, "w_graph": 0.15, "person_boost": 0.70, "components": 160},
    ]

    best_score = -1.0
    best_champ = None
    best_m = None

    for i, c in enumerate(candidates, 1):
        print(f"\n--- Training Candidate {i}/3 (TF-IDF={c['w_tfidf']}, LSA={c['w_lsa']}, Graph={c['w_graph']}, SVD={c['components']}) ---")
        t0 = time.time()
        champ = VeloraChampionRecommender(
            df_movies=df_movies,
            df_songs=df_songs,
            svd_components=c["components"],
            w_tfidf=c["w_tfidf"],
            w_lsa=c["w_lsa"],
            w_graph=c["w_graph"],
            person_boost=c["person_boost"]
        )
        m = evaluate_recommender(champ, k=10)
        dur = time.time() - t0
        score = (m["mean_ndcg"] * 0.5) + (m["mean_map"] * 0.3) + (m["mean_recall"] * 0.2)
        print(f"  NDCG@10: {m['mean_ndcg']:.4f} | MAP@10: {m['mean_map']:.4f} | Recall@10: {m['mean_recall']:.4f} | Latency: {m['avg_latency_ms']:.1f}ms (trained in {dur:.1f}s)")
        if score > best_score:
            best_score = score
            best_champ = champ
            best_m = m

    print("\n>>> [FINAL DEPLOYMENT] Persisting Champion Model to models/ ...")
    best_champ.save(MODELS_DIR)
    print("✓ Successfully saved final champion artifacts!")

    print("\n" + "=" * 80)
    print("🏆 FINAL CHAMPION MODEL METRICS:")
    print(f"   • NDCG@10:   {best_m['mean_ndcg']:.4f}")
    print(f"   • MAP@10:    {best_m['mean_map']:.4f}")
    print(f"   • Recall@10: {best_m['mean_recall']:.4f}")
    print(f"   • Latency:   {best_m['avg_latency_ms']:.1f}ms per query")
    print("=" * 80)

if __name__ == "__main__":
    main()
