# Cinevibe AI — Cyber-Cinema & Music Intelligence Engine

> **Intelligent Multi-Modal Entertainment Discovery Platform**  
> AI-powered recommender connecting Indian cinema, songs, artists, and storylines spanning 1950s classics to 2026 latest releases.

---

## ✨ Features

- **🎬 Cross-Domain Entertainment Discovery**: Unified hybrid recommendations across Bollywood, Pan-Indian cinema (Telugu, Tamil, Malayalam, Kannada), Marathi, and Gujarati films.
- **🎵 Comprehensive Music Universe**: Spanning Indian Independent (Indie), Bollywood melodies, Marathi hits, South Indian viral chartbusters, and Global English Pop.
- **⚡ Combinatorial Multi-Genre Matching**: Filter and discover titles matching multiple genres simultaneously (e.g. *Action + Thriller*, *Comedy + Horror*, or *Indian Indie + Soulful*).
- **📅 Historical & Release Year Filters**: Precise era presets (*2024–2026 Latest Releases*, *2020–2023 Modern Hits*, *90s Golden Era*, *Retro*, etc.) and exact release year filtering.
- **🕸️ Relational Knowledge Graph**: Entity chaining engine mapping creative connections between Actors, Filmmakers, Playback Singers, and Music Directors.
- **📺 Watch & Streaming Dispatcher**: Instant deep-links to verified platforms (JioHotstar, Netflix, Prime Video, JioCinema, YouTube Movies, Spotify, Apple Music, and JioSaavn).
- **⭐ Audience Vibe Rating Meter**: Dynamic 5-tier audience review meter (*Bad, OK-ish, Good, Go for it, Must Watch / Must Listen*).

---

## 🛠️ Technology Stack
## How the Recommendation System Works

Cinevibe AI uses a content-based recommendation approach to help users discover movies and music based on their preferences.

The system uses TF-IDF vectorization and cosine similarity to identify similarities between entertainment items.

**Recommendation Workflow:**

1. **User Input:** Users select their preferred genres, release years, or other available filters.
2. **Data Processing:** The system uses the movie and music datasets to find relevant entertainment content.
3. **Similarity Matching:** TF-IDF and cosine similarity help identify items with similar characteristics.
4. **Recommendations:** The application displays matching movies or songs for users to explore.
5. **Further Discovery:** Users can explore related artists and access available streaming platforms.

- **Backend**: Python 3, Flask REST API
- **Machine Learning**: Scikit-Learn (`TfidfVectorizer`, Cosine Similarity), NumPy, Pandas
- **Frontend**: Vanilla HTML5, Cyber-Cinema Glassmorphic CSS3, Vanilla ES6+ JavaScript (zero external JS dependencies)
- **Voice Search**: Browser-native Web Speech API
- **Data Architecture**: Curated multidimensional CSV datasets (3,017 movies and 11,795 songs)

---

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaheshaan2266-engg/ai-entertainment-recommender.git
   cd ai-entertainment-recommender
   ```

2. **Set up virtual environment & dependencies:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Train the AI Recommender models (optional, precomputed artifacts included):**
   ```bash
   python scripts/train_recommender.py
   ```

4. **Launch the web application:**
   ```bash
   python app.py
   ```

5. **Open in browser:**
   Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 Project Structure

```
├── ai_entertainment_recommender/  # Core package (Recommender, Graph Engine, Watch Provider, Loader)
├── data/
│   ├── bollywood/                # Enriched movies (3,017) and songs (11,795) datasets
│   ├── movie/                    # Metadata & ratings
│   └── music/                    # Music features
├── models/                       # Persisted TF-IDF vectorizers & feature matrices
├── scripts/                      # Training & dataset enrichment pipelines
├── static/
│   ├── css/style.css             # Cyber-Cinema design system & animations
│   ├── js/app.js                 # Reactive frontend logic & filter intelligence
│   └── images/                   # High-res poster assets & polaroid artwork
├── templates/
│   └── index.html                # Responsive web app template
├── app.py                        # Flask server & REST API
└── requirements.txt              # Project dependencies
```
