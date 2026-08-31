# AI Entertainment Recommender

This project contains starter data and recommender experiments for movies and music. It is set up as a small Python project so you can load the datasets, prototype recommendation logic, and extend the app with new models.

## Project structure

- `data/movie/` – movie ratings and metadata
- `data/music/` – music feature dataset
- `ai_entertainment_recommender/` – reusable Python package
- `movie_recommender/` – legacy movie scripts
- `music recommender/` – legacy music scripts
- `main.py` – project entry point

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

## Notes

The older scripts in the `movie_recommender/` and `music recommender/` directories were kept for compatibility, but the new package-style code is the recommended place to build and extend the project.
