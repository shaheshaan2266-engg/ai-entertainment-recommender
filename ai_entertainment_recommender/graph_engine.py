"""
Entertainment Knowledge Graph and Relational Chaining Engine.
Enables multi-hop traversal across Movies, Directors, Actors, Songs, Composers, and Singers.
"""

from collections import defaultdict
import re
import pandas as pd


def normalize_name(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    name = str(name).strip()
    return re.sub(r'\s+', ' ', name)


def split_names(text: str) -> list[str]:
    if not text or pd.isna(text):
        return []
    # Split by comma, pipe, or ' and '
    cleaned = str(text).replace('|', ',').replace(' / ', ',').replace(';', ',')
    parts = [normalize_name(p) for p in cleaned.split(',')]
    return [p for p in parts if p and p.lower() not in ['unknown', 'various', 'nan', 'none']]


class EntertainmentGraph:
    """
    In-memory bipartite and relational graph connecting:
    Movie <-> Director
    Movie <-> Actor
    Movie <-> Song
    Song  <-> Composer
    Song  <-> Singer
    """

    def __init__(self, df_movies: pd.DataFrame = None, df_songs: pd.DataFrame = None):
        self.movie_to_directors = defaultdict(set)
        self.director_to_movies = defaultdict(set)
        
        self.movie_to_actors = defaultdict(set)
        self.actor_to_movies = defaultdict(set)
        
        self.movie_to_songs = defaultdict(list)
        self.song_to_movie = {}
        
        self.song_to_composers = defaultdict(set)
        self.composer_to_songs = defaultdict(list)
        
        self.song_to_singers = defaultdict(set)
        self.singer_to_songs = defaultdict(list)

        self.actor_director_collabs = defaultdict(int)
        self.composer_singer_collabs = defaultdict(int)

        self.movies_by_title = {}
        self.songs_by_title = {}

        if df_movies is not None and df_songs is not None:
            self.build_graph(df_movies, df_songs)

    def build_graph(self, df_movies: pd.DataFrame, df_songs: pd.DataFrame):
        """Populates all graph edges from movies and songs dataframes."""
        # 1. Movies & People
        for _, row in df_movies.iterrows():
            title = normalize_name(row.get('title', ''))
            if not title:
                continue
            
            movie_dict = row.to_dict()
            self.movies_by_title[title.lower()] = movie_dict

            directors = split_names(row.get('director', ''))
            actors = split_names(row.get('actors', ''))

            for d in directors:
                d_clean = d.title()
                self.movie_to_directors[title].add(d_clean)
                self.director_to_movies[d_clean.lower()].add(title)

            for a in actors:
                a_clean = a.title()
                self.movie_to_actors[title].add(a_clean)
                self.actor_to_movies[a_clean.lower()].add(title)

            for d in directors:
                for a in actors:
                    self.actor_director_collabs[(a.title(), d.title())] += 1

        # 2. Songs & Musicians
        for _, row in df_songs.iterrows():
            song_title = normalize_name(row.get('title', ''))
            movie_title = normalize_name(row.get('movie_title', ''))
            if not song_title:
                continue

            song_dict = row.to_dict()
            self.songs_by_title[song_title.lower()] = song_dict

            if movie_title:
                self.movie_to_songs[movie_title.lower()].append(song_dict)
                self.song_to_movie[song_title.lower()] = movie_title

            singers = split_names(row.get('singer', ''))
            composers = split_names(row.get('composer', ''))

            for s in singers:
                s_clean = s.title()
                self.song_to_singers[song_title].add(s_clean)
                self.singer_to_songs[s_clean.lower()].append(song_dict)

            for c in composers:
                c_clean = c.title()
                self.song_to_composers[song_title].add(c_clean)
                self.composer_to_songs[c_clean.lower()].append(song_dict)

            for c in composers:
                for s in singers:
                    self.composer_singer_collabs[(c.title(), s.title())] += 1

    def crawl_chain(self, entity_type: str, entity_name: str, max_items: int = 15) -> dict:
        """
        Crawls and expands relational chains from a seed entity.
        entity_type: 'movie', 'song', 'actor', 'director', 'singer', 'composer'
        """
        entity_name_clean = normalize_name(entity_name)
        key = entity_name_clean.lower()
        result = {
            "seed": entity_name_clean,
            "type": entity_type,
            "movies": [],
            "songs": [],
            "directors": [],
            "actors": [],
            "singers": [],
            "composers": [],
            "collaborations": []
        }

        if entity_type == 'movie':
            movie_data = self.movies_by_title.get(key)
            if not movie_data:
                # Fuzzy fallback
                for k, v in self.movies_by_title.items():
                    if key in k:
                        movie_data = v
                        break

            if movie_data:
                m_title = movie_data['title']
                result["details"] = movie_data
                directors = list(self.movie_to_directors.get(m_title, []))
                actors = list(self.movie_to_actors.get(m_title, []))
                result["directors"] = directors
                result["actors"] = actors

                # Chained movies from same director(s)
                chained_movies = []
                for d in directors:
                    for om in self.director_to_movies.get(d.lower(), []):
                        if om.lower() != key and om.lower() in self.movies_by_title:
                            item = self.movies_by_title[om.lower()]
                            chained_movies.append({
                                **item,
                                "relation": f"Directed by {d}"
                            })

                # Chained movies from same actors
                for a in actors[:4]:
                    for om in list(self.actor_to_movies.get(a.lower(), []))[:6]:
                        if om.lower() != key and om.lower() in self.movies_by_title:
                            if not any(cm['title'] == om for cm in chained_movies):
                                item = self.movies_by_title[om.lower()]
                                chained_movies.append({
                                    **item,
                                    "relation": f"Stars {a}"
                                })

                result["movies"] = chained_movies[:max_items]

                # Chained songs from this movie
                movie_songs = self.movie_to_songs.get(key, [])
                if not movie_songs:
                    # check title match
                    movie_songs = self.movie_to_songs.get(m_title.lower(), [])
                result["songs"] = movie_songs[:max_items]

                # Composers and Singers from those songs
                for s in movie_songs:
                    composers = split_names(s.get('composer', ''))
                    singers = split_names(s.get('singer', ''))
                    result["composers"].extend(composers)
                    result["singers"].extend(singers)

                result["composers"] = list(dict.fromkeys(result["composers"]))
                result["singers"] = list(dict.fromkeys(result["singers"]))

        elif entity_type == 'song':
            song_data = self.songs_by_title.get(key)
            if not song_data:
                for k, v in self.songs_by_title.items():
                    if key in k:
                        song_data = v
                        break

            if song_data:
                s_title = song_data['title']
                result["details"] = song_data
                singers = list(self.song_to_singers.get(s_title, []))
                composers = list(self.song_to_composers.get(s_title, []))
                result["singers"] = singers
                result["composers"] = composers

                # Linked Movie
                m_title = song_data.get('movie_title', '')
                if not m_title:
                    m_title = self.song_to_movie.get(s_title.lower(), '')

                movie_item = None
                if m_title:
                    m_lower = m_title.lower()
                    if m_lower in self.movies_by_title:
                        movie_item = self.movies_by_title[m_lower]
                    else:
                        for mk, mv in self.movies_by_title.items():
                            if m_lower in mk or mk in m_lower:
                                movie_item = mv
                                break

                if movie_item:
                    result["movies"].append({
                        **movie_item,
                        "relation": f"Official Soundtrack of '{movie_item['title']}'"
                    })
                    result["directors"] = list(self.movie_to_directors.get(movie_item['title'], []))
                    result["actors"] = list(self.movie_to_actors.get(movie_item['title'], []))

                # Other songs from same movie album
                if m_title:
                    other_movie_songs = [
                        s for s in self.movie_to_songs.get(m_title.lower(), [])
                        if s.get('title', '').lower() != s_title.lower()
                    ]
                    for os in other_movie_songs[:6]:
                        result["songs"].append({
                            **os,
                            "relation": f"Soundtrack from '{m_title}'"
                        })

                # Chained songs from same composer(s)
                for c in composers:
                    for cs in self.composer_to_songs.get(c.lower(), [])[:6]:
                        if cs.get('title', '').lower() != s_title.lower() and not any(x.get('title') == cs.get('title') for x in result["songs"]):
                            result["songs"].append({
                                **cs,
                                "relation": f"Composed by {c}"
                            })

                # Chained songs from same singer(s)
                for s in singers:
                    for ss in self.singer_to_songs.get(s.lower(), [])[:6]:
                        if ss.get('title', '').lower() != s_title.lower() and not any(x.get('title') == ss.get('title') for x in result["songs"]):
                            result["songs"].append({
                                **ss,
                                "relation": f"Vocals by {s}"
                            })

                result["songs"] = result["songs"][:max_items]

                # Collaborations (Composer & Singer)
                for c in composers:
                    for s in singers:
                        pair_count = self.composer_singer_collabs.get((c.title(), s.title()), 1)
                        result["collaborations"].append({
                            "name": f"{s} & {c}",
                            "count": pair_count,
                            "role": "Composer-Singer Duo"
                        })

        elif entity_type == 'director':
            matched_key = key
            if key not in self.director_to_movies:
                for dk in self.director_to_movies:
                    if key in dk:
                        matched_key = dk
                        break
            movies = [self.movies_by_title[m.lower()] for m in self.director_to_movies.get(matched_key, []) if m.lower() in self.movies_by_title]
            result["movies"] = sorted(movies, key=lambda x: x.get('release_year', 0), reverse=True)[:max_items]
            
            # Find top actors collaborated with
            actor_counts = defaultdict(int)
            for m in movies:
                for a in split_names(m.get('actors', '')):
                    actor_counts[a] += 1
            top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:6]
            result["actors"] = [a[0] for a in top_actors]
            result["collaborations"] = [{"name": a[0], "count": a[1], "role": "Actor"} for a in top_actors]

            # Find soundtrack songs
            soundtrack_songs = []
            for m in movies:
                soundtrack_songs.extend(self.movie_to_songs.get(m['title'].lower(), []))
            result["songs"] = soundtrack_songs[:max_items]

        elif entity_type == 'actor':
            matched_key = key
            if key not in self.actor_to_movies:
                for ak in self.actor_to_movies:
                    if key in ak:
                        matched_key = ak
                        break
            movies = [self.movies_by_title[m.lower()] for m in self.actor_to_movies.get(matched_key, []) if m.lower() in self.movies_by_title]
            result["movies"] = sorted(movies, key=lambda x: x.get('release_year', 0), reverse=True)[:max_items]

            # Frequent directors
            director_counts = defaultdict(int)
            for m in movies:
                for d in split_names(m.get('director', '')):
                    director_counts[d] += 1
            top_dirs = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            result["directors"] = [d[0] for d in top_dirs]
            result["collaborations"] = [{"name": d[0], "count": d[1], "role": "Director"} for d in top_dirs]

            # Songs from their movies
            actor_songs = []
            for m in movies:
                actor_songs.extend(self.movie_to_songs.get(m['title'].lower(), []))
            result["songs"] = actor_songs[:max_items]

        elif entity_type == 'singer':
            matched_key = key
            if key not in self.singer_to_songs:
                for sk in self.singer_to_songs:
                    if key in sk:
                        matched_key = sk
                        break
            songs = self.singer_to_songs.get(matched_key, [])
            result["songs"] = songs[:max_items]

            # Composers & Movies
            composer_counts = defaultdict(int)
            movie_titles = set()
            for s in songs:
                m_title = s.get('movie_title', '')
                if m_title and m_title.lower() in self.movies_by_title:
                    movie_titles.add(m_title.lower())
                for c in split_names(s.get('composer', '')):
                    composer_counts[c] += 1

            top_composers = sorted(composer_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            result["composers"] = [c[0] for c in top_composers]
            result["collaborations"] = [{"name": c[0], "count": c[1], "role": "Composer"} for c in top_composers]
            result["movies"] = [self.movies_by_title[m] for m in list(movie_titles)[:max_items]]

        elif entity_type == 'composer':
            matched_key = key
            if key not in self.composer_to_songs:
                for ck in self.composer_to_songs:
                    if key in ck:
                        matched_key = ck
                        break
            songs = self.composer_to_songs.get(matched_key, [])
            result["songs"] = songs[:max_items]

            # Frequent singers & movies
            singer_counts = defaultdict(int)
            movie_titles = set()
            for s in songs:
                m_title = s.get('movie_title', '')
                if m_title and m_title.lower() in self.movies_by_title:
                    movie_titles.add(m_title.lower())
                for s_name in split_names(s.get('singer', '')):
                    singer_counts[s_name] += 1

            top_singers = sorted(singer_counts.items(), key=lambda x: x[1], reverse=True)[:6]
            result["singers"] = [s[0] for s in top_singers]
            result["collaborations"] = [{"name": s[0], "count": s[1], "role": "Singer"} for s in top_singers]
            result["movies"] = [self.movies_by_title[m] for m in list(movie_titles)[:max_items]]

        return result

    def compute_movie_chain_affinity(self, movie_a: dict, movie_b: dict) -> float:
        """
        Calculates graph affinity score (0.0 to 1.0) between two movies based on
        shared directors, actors, and collaboration frequency.
        """
        if not movie_a or not movie_b:
            return 0.0
        
        score = 0.0
        dir_a = set(split_names(movie_a.get('director', '')))
        dir_b = set(split_names(movie_b.get('director', '')))
        shared_dirs = dir_a.intersection(dir_b)
        if shared_dirs:
            score += 0.40

        act_a = set(split_names(movie_a.get('actors', '')))
        act_b = set(split_names(movie_b.get('actors', '')))
        shared_acts = act_a.intersection(act_b)
        if shared_acts:
            score += min(0.40, len(shared_acts) * 0.20)

        # Cross director-actor history
        for a in act_a:
            for d in dir_b:
                if self.actor_director_collabs.get((a.title(), d.title()), 0) > 1:
                    score += 0.10
                    break

        return min(1.0, score)
