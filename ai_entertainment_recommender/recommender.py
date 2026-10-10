"""
Unified Multi-Modal Entertainment Recommender for Indian Cinema and Music.
Combines TF-IDF feature fusion, relational graph chaining, and explainable AI.
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ai_entertainment_recommender.data_loader import load_bollywood_movies, load_bollywood_songs
from ai_entertainment_recommender.graph_engine import EntertainmentGraph, normalize_name, split_names


ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedEntertainmentRecommender:
    """
    Unified AI Recommender supporting:
    - Mode: 'movies'
    - Mode: 'songs'
    - Mode: 'both'
    """

    def __init__(self, df_movies: pd.DataFrame = None, df_songs: pd.DataFrame = None, auto_train: bool = True, force_retrain: bool = False):
        self.df_movies = df_movies if df_movies is not None else load_bollywood_movies()
        self.df_songs = df_songs if df_songs is not None else load_bollywood_songs()

        # Build Graph
        self.graph = EntertainmentGraph(self.df_movies, self.df_songs)

        self.movie_vectorizer = None
        self.song_vectorizer = None
        self.movie_tfidf_matrix = None
        self.song_tfidf_matrix = None

        if auto_train:
            self._fit_or_load_models(force_retrain=force_retrain)

    def _prepare_movie_features(self) -> list[str]:
        features = []
        for _, row in self.df_movies.iterrows():
            title = str(row.get('title', ''))
            genres = str(row.get('genres', '')).replace(',', ' ')
            director = str(row.get('director', ''))
            actors = str(row.get('actors', '')).replace(',', ' ')
            overview = str(row.get('overview', ''))
            era = str(row.get('era', ''))
            industry = str(row.get('industry', ''))
            year = str(row.get('release_year', ''))

            # Weight important entity fields by repeating them
            combined = f"{title} {title} {genres} {genres} {genres} {director} {director} {actors} {actors} {overview} {era} {industry} {industry} {year}"
            features.append(combined.lower())
        return features

    def _prepare_song_features(self) -> list[str]:
        features = []
        for _, row in self.df_songs.iterrows():
            title = str(row.get('title', ''))
            movie = str(row.get('movie_title', ''))
            singer = str(row.get('singer', '')).replace(',', ' ')
            composer = str(row.get('composer', ''))
            lyricist = str(row.get('lyricist', ''))
            mood = str(row.get('mood', '')).replace(',', ' ')
            era = str(row.get('era', ''))
            year = str(row.get('release_year', ''))
            snippet = str(row.get('lyrics_snippet', ''))

            combined = f"{title} {title} {movie} {movie} {singer} {singer} {composer} {composer} {lyricist} {mood} {mood} {mood} {era} {year} {snippet}"
            features.append(combined.lower())
        return features

    def _fit_or_load_models(self, force_retrain: bool = False):
        movie_vec_path = MODELS_DIR / "movie_vectorizer.pkl"
        movie_mat_path = MODELS_DIR / "movie_tfidf.pkl"
        song_vec_path = MODELS_DIR / "song_vectorizer.pkl"
        song_mat_path = MODELS_DIR / "song_tfidf.pkl"

        if not force_retrain and (movie_vec_path.exists() and movie_mat_path.exists() and 
            song_vec_path.exists() and song_mat_path.exists()):
            try:
                with open(movie_vec_path, "rb") as f:
                    self.movie_vectorizer = pickle.load(f)
                with open(movie_mat_path, "rb") as f:
                    self.movie_tfidf_matrix = pickle.load(f)
                with open(song_vec_path, "rb") as f:
                    self.song_vectorizer = pickle.load(f)
                with open(song_mat_path, "rb") as f:
                    self.song_tfidf_matrix = pickle.load(f)
                return
            except Exception:
                pass

        # Intensive TF-IDF training on fused multi-modal representations
        movie_texts = self._prepare_movie_features()
        self.movie_vectorizer = TfidfVectorizer(
            max_features=18000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.movie_tfidf_matrix = self.movie_vectorizer.fit_transform(movie_texts)

        song_texts = self._prepare_song_features()
        self.song_vectorizer = TfidfVectorizer(
            max_features=25000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.song_tfidf_matrix = self.song_vectorizer.fit_transform(song_texts)

        # Save models
        with open(movie_vec_path, "wb") as f:
            pickle.dump(self.movie_vectorizer, f)
        with open(movie_mat_path, "wb") as f:
            pickle.dump(self.movie_tfidf_matrix, f)
        with open(song_vec_path, "wb") as f:
            pickle.dump(self.song_vectorizer, f)
        with open(song_mat_path, "wb") as f:
            pickle.dump(self.song_tfidf_matrix, f)

    def search_autocomplete(self, query: str, limit: int = 8) -> list[dict]:
        """Provides instant typeahead across movies, songs, actors, directors, singers, composers."""
        if not query or len(query.strip()) < 2:
            return []

        q = query.strip().lower()
        results = []
        seen = set()

        # 1. Check Movies
        for _, row in self.df_movies.iterrows():
            title = str(row.get('title') or '')
            if q in title.lower() and ('movie', title.lower()) not in seen:
                seen.add(('movie', title.lower()))
                genre_first = str(row.get('genres') or 'Cinema').split(',')[0].strip()
                results.append({
                    "title": title,
                    "type": "Movie",
                    "badge": f"{row.get('release_year', '')} · {genre_first}",
                    "thumbnail": str(row.get('poster_url') or '')
                })
                if len(results) >= limit:
                    return results

        # 2. Check Songs
        for _, row in self.df_songs.iterrows():
            title = str(row.get('title') or '')
            if q in title.lower() and ('song', title.lower()) not in seen:
                seen.add(('song', title.lower()))
                singer_first = str(row.get('singer') or 'Various').split(',')[0].strip()
                movie_name = str(row.get('movie_title') or '')
                results.append({
                    "title": title,
                    "type": "Song",
                    "badge": f"{singer_first} · {movie_name}" if movie_name else singer_first,
                    "thumbnail": str(row.get('thumbnail_url') or '')
                })
                if len(results) >= limit:
                    return results

        # 3. Check Directors & Actors
        for _, row in self.df_movies.iterrows():
            director = str(row.get('director') or '')
            if q in director.lower() and ('director', director.lower()) not in seen:
                seen.add(('director', director.lower()))
                results.append({
                    "title": director,
                    "type": "Director",
                    "badge": "Filmmaker",
                    "thumbnail": ""
                })
            for a in split_names(row.get('actors', '')):
                if q in a.lower() and ('actor', a.lower()) not in seen:
                    seen.add(('actor', a.lower()))
                    results.append({
                        "title": a,
                        "type": "Actor",
                        "badge": "Star Cast",
                        "thumbnail": ""
                    })
                if len(results) >= limit:
                    return results

        # 4. Check Singers & Composers
        for _, row in self.df_songs.iterrows():
            for s in split_names(row.get('singer', '')):
                if q in s.lower() and ('singer', s.lower()) not in seen:
                    seen.add(('singer', s.lower()))
                    results.append({
                        "title": s,
                        "type": "Singer",
                        "badge": "Playback Artist",
                        "thumbnail": ""
                    })
            for c in split_names(row.get('composer', '')):
                if q in c.lower() and ('composer', c.lower()) not in seen:
                    seen.add(('composer', c.lower()))
                    results.append({
                        "title": c,
                        "type": "Composer",
                        "badge": "Music Director",
                        "thumbnail": ""
                    })
                if len(results) >= limit:
                    return results

        return results[:limit]

    def _generate_movie_explanation(self, source_meta: dict, target_row: pd.Series, matched_person: dict = None, matched_genres: list = None) -> str:
        """Constructs an explainable rationale badge for a movie recommendation."""
        if matched_genres:
            genres_label = " + ".join([g.title() for g in matched_genres[:2]])
            return f"Matching {genres_label} · {target_row.get('release_year', '')} · {target_row.get('director', '')}"

        if matched_person:
            p_type = matched_person.get('type')
            p_name = matched_person.get('title')
            if p_type == 'Director' and p_name.lower() in str(target_row.get('director', '')).lower():
                return f"Directed by {p_name}"
            if p_type == 'Actor' and p_name.lower() in str(target_row.get('actors', '')).lower():
                return f"Starring {p_name}"
            if p_type in ['Singer', 'Composer']:
                return f"Features music by {p_name}"

        reasons = []
        if source_meta:
            s_dir = set(split_names(source_meta.get('director', '')))
            t_dir = set(split_names(target_row.get('director', '')))
            shared_dirs = s_dir.intersection(t_dir)
            if shared_dirs:
                reasons.append(f"Directed by {list(shared_dirs)[0]}")

            s_acts = set(split_names(source_meta.get('actors', '')))
            t_acts = set(split_names(target_row.get('actors', '')))
            shared_acts = s_acts.intersection(t_acts)
            if shared_acts:
                reasons.append(f"Stars {list(shared_acts)[0]}")

            s_genres = set([g.strip().lower() for g in str(source_meta.get('genres', '')).split(',') if g.strip()])
            t_genres = set([g.strip().lower() for g in str(target_row.get('genres', '')).split(',') if g.strip()])
            shared_g = s_genres.intersection(t_genres)
            if shared_g and len(reasons) < 2:
                reasons.append(f"{list(shared_g)[0].title()} Theme")

            if source_meta.get('era') == target_row.get('era') and len(reasons) < 2:
                reasons.append(f"{target_row.get('era')}")

        if not reasons:
            first_genre = str(target_row.get('genres', 'Cinema')).split(',')[0].strip()
            reasons.append(f"{first_genre} Masterpiece · {target_row.get('release_year', '')}")

        return " · ".join(reasons)

    def _generate_song_explanation(self, source_meta: dict, target_row: pd.Series, matched_person: dict = None, matched_genres: list = None) -> str:
        """Constructs an explainable rationale badge for a song recommendation."""
        if matched_genres:
            genres_label = " + ".join([g.title() for g in matched_genres[:2]])
            return f"Matching {genres_label} · {target_row.get('movie_title', 'Music')}"

        if matched_person:
            p_type = matched_person.get('type')
            p_name = matched_person.get('title')
            if p_type == 'Singer' and p_name.lower() in str(target_row.get('singer', '')).lower():
                return f"Vocals by {p_name}"
            if p_type == 'Composer' and p_name.lower() in str(target_row.get('composer', '')).lower():
                return f"Composed by {p_name}"
            if p_type in ['Actor', 'Director']:
                return f"From {p_name}'s filmography"

        reasons = []
        if source_meta:
            s_singer = set(split_names(source_meta.get('singer', '')))
            t_singer = set(split_names(target_row.get('singer', '')))
            shared_s = s_singer.intersection(t_singer)
            if shared_s:
                reasons.append(f"Vocals by {list(shared_s)[0]}")

            s_comp = set(split_names(source_meta.get('composer', '')))
            t_comp = set(split_names(target_row.get('composer', '')))
            shared_c = s_comp.intersection(t_comp)
            if shared_c:
                reasons.append(f"Composed by {list(shared_c)[0]}")

            if source_meta.get('mood') == target_row.get('mood'):
                reasons.append(f"{target_row.get('mood')}")

        if not reasons:
            reasons.append(f"{target_row.get('mood', 'Hit Melodies')} · from {target_row.get('movie_title', 'Soundtrack')}")

        return " · ".join(reasons)

    def get_filter_options(self) -> dict:
        """Returns structured genres, years, and era presets for UI search bar filters."""
        movie_genres = [
            "Action", "Drama", "Comedy", "Romance", "Thriller", "Crime", 
            "Sci-Fi", "Horror", "Mystery", "Historical", "Biography", 
            "Fantasy", "Musical", "Family", "Adventure", "War", "Sport", "Epic"
        ]
        
        music_genres = [
            "Romantic & Soulful", "Bollywood Melodies", "Indian Indie", 
            "Pop & Dance", "Hip Hop / Rap", "Sufi & Spiritual", 
            "Classical & Semi-Classical", "Rock & Alternative", 
            "Marathi & Folk", "Global / English Pop"
        ]
        
        both_genres = [
            "Action & High Energy", "Romantic & Soulful", "Comedy & Fun", 
            "Drama & Emotional", "Thriller & Mystery", "Classical & Heritage", 
            "Indian Indie & Folk", "Global Hits", "Epic & Historical", "Dance Party"
        ]

        year_presets = [
            {"label": "All Years", "value": "All"},
            {"label": "2024–2026 (Latest Releases)", "value": "2024-2026"},
            {"label": "2020–2023 (Modern Hits)", "value": "2020-2023"},
            {"label": "2010–2019 (2010s Blockbusters)", "value": "2010-2019"},
            {"label": "2000–2009 (Millennial Classics)", "value": "2000-2009"},
            {"label": "1990–1999 (90s Golden Era)", "value": "1990-1999"},
            {"label": "1970–1989 (Retro Vintage)", "value": "1970-1989"},
            {"label": "Pre-1970 (Vintage Golden Age)", "value": "1930-1969"}
        ]

        distinct_years = []
        for y in self.df_movies['release_year'].dropna().unique():
            try:
                distinct_years.append(int(float(y)))
            except (ValueError, TypeError):
                continue
        years = sorted(list(set(distinct_years)), reverse=True)

        return {
            "movie_genres": movie_genres,
            "music_genres": music_genres,
            "both_genres": both_genres,
            "year_presets": year_presets,
            "years": years
        }

    def recommend(
        self,
        query: str = "",
        mode: str = "both",
        era_filter: str = None,
        genre_filter: str = None,
        genres_filter: list = None,
        year_filter: str = None,
        top_n: int = 12
    ) -> dict:
        """
        Executes unified hybrid recommendation with multi-genre combinations and year filtering.
        mode: 'movies', 'songs', 'both'
        """
        query_clean = str(query or "").strip()
        q_lower = query_clean.lower()

        # Parse selected genres (supports list, comma-separated string, or legacy single genre)
        selected_genres = []
        if genres_filter:
            if isinstance(genres_filter, str):
                selected_genres = [g.strip() for g in genres_filter.split(",") if g.strip() and g.strip().lower() != "all"]
            elif isinstance(genres_filter, list):
                selected_genres = [str(g).strip() for g in genres_filter if g and str(g).strip().lower() != "all"]
        elif genre_filter and str(genre_filter).strip().lower() != "all":
            selected_genres = [genre_filter.strip()]

        def _matches_year(val) -> bool:
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
            if yf.endswith("+"):
                try:
                    return y >= int(yf[:-1])
                except ValueError:
                    pass
            try:
                return y == int(yf)
            except ValueError:
                pass
            return True

        def _match_genres(text: str) -> tuple[bool, int, list]:
            if not selected_genres:
                return True, 0, []
            t = str(text).lower()
            matched = []
            for g in selected_genres:
                clean_g = g.strip().lower()
                if not clean_g:
                    continue
                # Split compound genre names like "Pop & Dance" -> check either
                parts = [p.strip() for p in clean_g.replace('&', '/').split('/') if p.strip()]
                if any(p in t for p in parts) or clean_g in t:
                    matched.append(g)
            return (len(matched) > 0), len(matched), matched

        # Synthesize effective text query if user selected filters without typing a title
        effective_query = query_clean
        if not effective_query and selected_genres:
            effective_query = " ".join(selected_genres)
            if year_filter and str(year_filter) != "All":
                effective_query += " " + str(year_filter)
        elif not effective_query:
            effective_query = "Bollywood Indian cinema music hits"

        matched_movie = None
        matched_song = None
        matched_person = None
        matched_movie_idx = None
        matched_song_idx = None

        if query_clean:
            # 1. Check exact movie
            for idx, row in self.df_movies.iterrows():
                if str(row.get('title', '')).lower() == q_lower:
                    matched_movie = row.to_dict()
                    matched_movie_idx = idx
                    break
            
            # 2. Check exact song
            if not matched_movie:
                for idx, row in self.df_songs.iterrows():
                    if str(row.get('title', '')).lower() == q_lower:
                        matched_song = row.to_dict()
                        matched_song_idx = idx
                        break

            # 3. Check exact Person (Singer, Composer, Actor, Director)
            if not matched_movie and not matched_song:
                if q_lower in self.graph.singer_to_songs:
                    matched_person = {
                        "type": "Singer",
                        "title": q_lower.title(),
                        "data": {"count": len(self.graph.singer_to_songs[q_lower]), "role": "Playback Singer"}
                    }
                elif q_lower in self.graph.composer_to_songs:
                    matched_person = {
                        "type": "Composer",
                        "title": q_lower.title(),
                        "data": {"count": len(self.graph.composer_to_songs[q_lower]), "role": "Music Composer"}
                    }
                elif q_lower in self.graph.actor_to_movies:
                    matched_person = {
                        "type": "Actor",
                        "title": q_lower.title(),
                        "data": {"count": len(self.graph.actor_to_movies[q_lower]), "role": "Actor / Star"}
                    }
                elif q_lower in self.graph.director_to_movies:
                    matched_person = {
                        "type": "Director",
                        "title": q_lower.title(),
                        "data": {"count": len(self.graph.director_to_movies[q_lower]), "role": "Filmmaker"}
                    }

            # 4. Check substring movie match if neither exact
            if not matched_movie and not matched_song and not matched_person:
                for idx, row in self.df_movies.iterrows():
                    if q_lower in str(row.get('title', '')).lower():
                        matched_movie = row.to_dict()
                        matched_movie_idx = idx
                        break

            # 5. Check substring song match
            if not matched_movie and not matched_song and not matched_person:
                for idx, row in self.df_songs.iterrows():
                    if q_lower in str(row.get('title', '')).lower():
                        matched_song = row.to_dict()
                        matched_song_idx = idx
                        break

            # 6. Check substring person match
            if not matched_movie and not matched_song and not matched_person:
                for sk, s_list in self.graph.singer_to_songs.items():
                    if q_lower in sk:
                        matched_person = {"type": "Singer", "title": sk.title(), "data": {"count": len(s_list), "role": "Playback Singer"}}
                        break
                if not matched_person:
                    for ck, c_list in self.graph.composer_to_songs.items():
                        if q_lower in ck:
                            matched_person = {"type": "Composer", "title": ck.title(), "data": {"count": len(c_list), "role": "Music Composer"}}
                            break
                if not matched_person:
                    for ak, a_list in self.graph.actor_to_movies.items():
                        if q_lower in ak:
                            matched_person = {"type": "Actor", "title": ak.title(), "data": {"count": len(a_list), "role": "Actor / Star"}}
                            break
                if not matched_person:
                    for dk, d_list in self.graph.director_to_movies.items():
                        if q_lower in dk:
                            matched_person = {"type": "Director", "title": dk.title(), "data": {"count": len(d_list), "role": "Filmmaker"}}
                            break

        recommendations = {
            "query": query_clean or ", ".join(selected_genres),
            "mode": mode,
            "filters_applied": {
                "genres": selected_genres,
                "year": year_filter or "All",
                "era": era_filter or "All"
            },
            "identified": None,
            "movies": [],
            "songs": []
        }

        if matched_movie:
            recommendations["identified"] = {
                "type": "Movie",
                "title": matched_movie['title'],
                "explanation": f"{matched_movie.get('release_year', '')} · {matched_movie.get('genres', '')} · Directed by {matched_movie.get('director', '')}",
                "data": matched_movie
            }
        elif matched_song:
            recommendations["identified"] = {
                "type": "Song",
                "title": matched_song['title'],
                "explanation": f"Vocals by {matched_song.get('singer', '')} · from {matched_song.get('movie_title', 'Soundtrack')}",
                "data": matched_song
            }
        elif matched_person:
            recommendations["identified"] = {
                "type": matched_person['type'],
                "title": matched_person['title'],
                "explanation": f"Mapped {matched_person['data'].get('role', 'Artist')} across Indian Entertainment Graph",
                "data": matched_person
            }

        recommendations["seed"] = recommendations["identified"]

        # ------------------- 1. MOVIE RECOMMENDATIONS -------------------
        if mode in ["movies", "both"]:
            if matched_movie_idx is not None:
                q_vec = self.movie_tfidf_matrix[matched_movie_idx]
            else:
                q_vec = self.movie_vectorizer.transform([effective_query.lower()])

            sim_scores = cosine_similarity(q_vec, self.movie_tfidf_matrix).flatten()
            
            combined_scores = []
            for idx, score in enumerate(sim_scores):
                if matched_movie_idx is not None and idx == matched_movie_idx:
                    continue

                row = self.df_movies.iloc[idx]

                # Year filter
                if not _matches_year(row.get('release_year')):
                    continue

                # Era filter
                if era_filter and era_filter != "All":
                    if row.get('era') != era_filter:
                        continue

                # Multi-genre combinations match
                movie_genre_str = f"{row.get('genres', '')} {row.get('overview', '')} {row.get('industry', '')}"
                is_g_matched, g_match_cnt, matched_genres_list = _match_genres(movie_genre_str)
                if selected_genres and not is_g_matched:
                    continue

                final_score = float(score)

                # Genre combination multi-match boost
                if selected_genres and is_g_matched:
                    ratio = g_match_cnt / len(selected_genres)
                    final_score += (0.45 * ratio)
                    if g_match_cnt == len(selected_genres):
                        final_score += 0.35  # Perfect multi-genre match bonus

                # Add Graph Affinity boost if seed movie exists
                if matched_movie:
                    affinity = self.graph.compute_movie_chain_affinity(matched_movie, row.to_dict())
                    final_score = (0.55 * final_score) + (0.45 * affinity)

                # Person boost
                if matched_person:
                    p_name = matched_person['title'].lower()
                    if matched_person['type'] == 'Director':
                        if p_name in str(row.get('director', '')).lower():
                            final_score += 0.50
                    elif matched_person['type'] == 'Actor':
                        if p_name in str(row.get('actors', '')).lower():
                            final_score += 0.50
                    elif matched_person['type'] in ['Singer', 'Composer']:
                        m_title = str(row.get('title', '')).lower()
                        if m_title in self.graph.movie_to_songs:
                            for ms in self.graph.movie_to_songs[m_title]:
                                if p_name in str(ms.get('singer', '')).lower() or p_name in str(ms.get('composer', '')).lower():
                                    final_score += 0.35
                                    break

                # Slight IMDB rating reinforcement
                try:
                    rating = float(row.get('imdb_rating', 7.0))
                    final_score += (rating - 7.0) * 0.05
                except (ValueError, TypeError):
                    pass

                combined_scores.append((idx, final_score, matched_genres_list))

            combined_scores.sort(key=lambda x: x[1], reverse=True)
            top_movie_indices = combined_scores[:top_n]

            movie_results = []
            for idx, score, m_genres in top_movie_indices:
                row = self.df_movies.iloc[idx]
                item = row.to_dict()
                item['similarity_score'] = round(float(score), 3)
                item['reason'] = self._generate_movie_explanation(matched_movie, row, matched_person=matched_person, matched_genres=m_genres if selected_genres else None)
                movie_results.append(item)

            recommendations["movies"] = movie_results

        # ------------------- 2. SONG RECOMMENDATIONS -------------------
        if mode in ["songs", "both"]:
            if matched_song_idx is not None:
                q_vec = self.song_tfidf_matrix[matched_song_idx]
            elif matched_movie:
                movie_context = f"{matched_movie.get('title')} {matched_movie.get('genres')} {matched_movie.get('actors')} {matched_movie.get('director')}"
                q_vec = self.song_vectorizer.transform([movie_context.lower()])
            else:
                q_vec = self.song_vectorizer.transform([effective_query.lower()])

            song_sim_scores = cosine_similarity(q_vec, self.song_tfidf_matrix).flatten()

            combined_song_scores = []
            for idx, score in enumerate(song_sim_scores):
                if matched_song_idx is not None and idx == matched_song_idx:
                    continue

                row = self.df_songs.iloc[idx]

                # Year filter
                if not _matches_year(row.get('release_year')):
                    continue

                # Era filter
                if era_filter and era_filter != "All":
                    if row.get('era') != era_filter:
                        continue

                # Genre matching
                song_genre_str = f"{row.get('mood', '')} {row.get('genres', '')} {row.get('lyrics_snippet', '')} {row.get('movie_title', '')}"
                is_g_matched, g_match_cnt, matched_genres_list = _match_genres(song_genre_str)

                # Cross-domain genre semantic mapping fallback if 0 direct matches
                if not is_g_matched and selected_genres:
                    synonyms = {
                        "action": ["bass boosted", "high energy", "fast", "dance", "desi beats"],
                        "romance": ["romantic", "soulful", "celebrated love anthem", "melancholy", "acoustic"],
                        "comedy": ["funny", "fast", "groovy", "dance", "celebration", "desi beats"],
                        "drama": ["emotional", "heartbreak", "soulful", "melancholy"],
                        "horror": ["mystery", "dark", "eerie", "bass"],
                        "historical": ["classical", "devotional", "epic", "folk", "heritage"],
                        "musical": ["melodies", "classical", "acoustic", "pop"],
                        "indie": ["indie & acoustic", "rock", "pop"],
                        "global": ["english pop", "global pop", "synth-pop"]
                    }
                    syn_cnt = 0
                    syn_matched = []
                    for g in selected_genres:
                        for mapped in synonyms.get(g.lower(), []):
                            if mapped in song_genre_str.lower():
                                syn_cnt += 1
                                syn_matched.append(g)
                                break
                    if syn_cnt > 0:
                        is_g_matched = True
                        g_match_cnt = syn_cnt
                        matched_genres_list = syn_matched

                if selected_genres and not is_g_matched:
                    continue

                final_score = float(score)

                # Genre combination multi-match boost
                if selected_genres and is_g_matched:
                    ratio = g_match_cnt / len(selected_genres)
                    final_score += (0.45 * ratio)
                    if g_match_cnt == len(selected_genres):
                        final_score += 0.35

                # Boost if song belongs to searched movie or related filmmaker
                if matched_movie:
                    movie_t = str(row.get('movie_title') or '').strip().lower()
                    target_t = str(matched_movie.get('title') or '').strip().lower()
                    if movie_t and target_t and (movie_t == target_t or movie_t in target_t or target_t in movie_t):
                        final_score += 0.40  # Official soundtrack boost!

                if matched_song:
                    s_singers = set(split_names(matched_song.get('singer', '')))
                    t_singers = set(split_names(row.get('singer', '')))
                    if s_singers.intersection(t_singers):
                        final_score += 0.25
                    
                    s_comp = set(split_names(matched_song.get('composer', '')))
                    t_comp = set(split_names(row.get('composer', '')))
                    if s_comp.intersection(t_comp):
                        final_score += 0.20

                # Person boost for songs
                if matched_person:
                    p_name = matched_person['title'].lower()
                    if matched_person['type'] == 'Singer':
                        if p_name in str(row.get('singer', '')).lower():
                            final_score += 0.55
                    elif matched_person['type'] == 'Composer':
                        if p_name in str(row.get('composer', '')).lower():
                            final_score += 0.55
                    elif matched_person['type'] in ['Actor', 'Director']:
                        m_title = str(row.get('movie_title', '')).lower()
                        if m_title in self.graph.movies_by_title:
                            m_info = self.graph.movies_by_title[m_title]
                            if p_name in str(m_info.get('actors', '')).lower() or p_name in str(m_info.get('director', '')).lower():
                                final_score += 0.35

                combined_song_scores.append((idx, final_score, matched_genres_list))

            combined_song_scores.sort(key=lambda x: x[1], reverse=True)
            top_song_indices = combined_song_scores[:top_n]

            song_results = []
            for idx, score, s_genres in top_song_indices:
                row = self.df_songs.iloc[idx]
                item = row.to_dict()
                item['similarity_score'] = round(float(score), 3)
                source_for_expl = matched_song or (matched_movie if matched_movie else None)
                item['reason'] = self._generate_song_explanation(source_for_expl, row, matched_person=matched_person, matched_genres=s_genres if selected_genres else None)
                song_results.append(item)

            recommendations["songs"] = song_results

        return recommendations

    def get_curated_trending(self) -> dict:
        """Returns thematic collections for instant landing discovery."""
        return {
            "pan_indian_blockbusters": self.df_movies[
                self.df_movies['title'].isin(['RRR', 'Baahubali 2: The Conclusion', 'KGF: Chapter 2', 'Kantara', 'Pushpa: The Rise', 'Kalki 2898 AD', 'Jailer', 'Vikram'])
            ].to_dict(orient='records'),
            
            "modern_bollywood": self.df_movies[
                self.df_movies['title'].isin(['Stree 2', 'Animal', 'Jawan', '12th Fail', 'Dunki', 'Fighter', 'Chhaava', 'Bhool Bhulaiyaa 3'])
            ].to_dict(orient='records'),

            "golden_classics": self.df_movies[
                self.df_movies['title'].isin(['Dilwale Dulhania Le Jayenge', '3 Idiots', 'Sholay', 'Lagaan', 'Rockstar', 'Kal Ho Naa Ho', 'Swades', 'Zindagi Na Milegi Dobara'])
            ].to_dict(orient='records'),

            "viral_hit_songs": self.df_songs[
                self.df_songs['title'].isin(['Naatu Naatu', 'Tauba Tauba', 'Aaj Ki Raat', 'Kesariya', 'Oo Antava Mava Oo Oo Antava', 'Apna Bana Le', 'Kaavaalaa', 'Pehle Bhi Main', 'Satranga', 'O Maahi', 'Kun Faya Kun', 'Tum Hi Ho', 'Agar Tum Saath Ho'])
            ].to_dict(orient='records')
        }
